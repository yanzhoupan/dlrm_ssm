#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/core/TensorAccessor.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <thrust/execution_policy.h>
#include <thrust/unique.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/host_vector.h>

#include <vector>

constexpr int MODE_SUM = 0;
constexpr int MODE_MEAN = 1;
constexpr int MODE_MAX = 2;

constexpr int NWEIGHT_PER_THREAD = 128;

// Fast ceil division (no overflow checking)
__host__ __device__ __forceinline__
int64_t ceil_div(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}

__global__
void krn_partials_per_segment(int64_t *ret, const int64_t *segment_offsets,
                              int64_t num_segments, int64_t numel) {
  const int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id < num_segments) {
    const int64_t idx_start = segment_offsets[id];
    const int64_t idx_end = (id == num_segments-1)?numel:segment_offsets[id+1];
    const int64_t size = idx_end - idx_start;
    ret[id] = ceil_div(size, NWEIGHT_PER_THREAD);
  }
}

__global__
void krn_partial_segment_offset(
        int64_t *ret,
        const int64_t *partials_per_segment,
        const int64_t *partials_per_segment_offset,
        const int64_t *segment_offsets,
        int64_t num_segments) {
  const int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id < num_segments) {
    int64_t idx = partials_per_segment_offset[id];
    const int64_t num_partials = partials_per_segment[id];
    const int64_t segment_offset = segment_offsets[id];
    for (int64_t i=0; i<num_partials; ++i) {
      ret[idx++] = segment_offset + i * NWEIGHT_PER_THREAD;
    }
  }
}


__device__ __host__ int64_t hash_func_backup(int64_t a, int64_t b) {
    return a + b;
}

__device__ __host__ int64_t hash_func(int64_t a, int64_t b) {
    return (a * 9824516537u + b * 57857966300227u) % 117130198221199u;
}

template<typename scalar_t>
__global__ void hashed_embedding_bag_update_output_kernel(
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> offsets,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> hashed_weights,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> output,
    torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> offset2bag,
    int64_t numIndices,
    int64_t numBags,
    int64_t embedding_dim,
    int64_t hashedWeightSize,
    int mode,
    torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> hashed_index,
    torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> bag_size,
    torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> max_indices)
{
    // the strategy here is that each bag x feature is handled by a single thread

    int64_t chunksPerBag = (embedding_dim + (int64_t)blockDim.x - 1) / (int64_t)blockDim.x;
    int64_t numChunks = numBags * chunksPerBag;
    int64_t chunkOffset = blockIdx.x * blockDim.y + threadIdx.y;
    int64_t chunkStride = gridDim.x * blockDim.y;

    for (int64_t chunk = chunkOffset; chunk < numChunks; chunk += chunkStride) {
        int64_t featureDim = (chunk % chunksPerBag) * blockDim.x + threadIdx.x;
        if (featureDim < embedding_dim) {
            int64_t bag = chunk / chunksPerBag;
            int64_t begin = bag == 0 ? 0 : offsets[bag]; // forces first offset to be 0 instead of asserting on it
            int64_t end = (bag < numBags - 1) ? (offsets[bag + 1]) : numIndices;
            CUDA_KERNEL_ASSERT(end >= begin);

            scalar_t weightFeatSum = 0;
            scalar_t weightFeatMax;

            int64_t bag_size_ = 0;
            int64_t maxWord = -1;
            // from start of bag to end of bag.
            for (int64_t emb = begin; emb < end; emb++) {
                const int64_t weightRow = input[emb];
                const int64_t hashedWeightIdx = hash_func(weightRow, featureDim) % hashedWeightSize;
                hashed_index[emb][featureDim] = hashedWeightIdx;
                scalar_t weightValue = hashed_weights[hashedWeightIdx];

                if (mode == MODE_MAX) {
                    if (emb == begin || weightValue > weightFeatMax) {
                        weightFeatMax = weightValue;
                        maxWord = input[emb];
                    }
                } else {
                    weightFeatSum += static_cast<scalar_t>(weightValue);
                }

                bag_size_++;
                if (featureDim == 0) {
                offset2bag[emb] = bag;
                }
            }
            if (mode == MODE_MEAN) {
                    if (end == begin) {
                    bag_size[bag] = 0;
                    } else {
                        weightFeatSum = weightFeatSum / static_cast<scalar_t>(bag_size_);
                        bag_size[bag] = bag_size_;
                    }
            }

            if (mode == MODE_MEAN || mode == MODE_SUM) {
                output[bag][featureDim] = static_cast<scalar_t>(weightFeatSum);
            }
            else if (mode == MODE_MAX) {
                if (end == begin) {
                    // If bag is empty, set output to 0.
                    weightFeatMax = 0;
                }
                max_indices[bag][featureDim] = maxWord;
                output[bag][featureDim] = weightFeatMax;
            }
        }
    }
}

template<typename scalar_t>
__global__ void compute_grad_weight_bags(
    torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> orig_hash_idx_idx,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> output_grad,
    torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> offset2bag,
    int64_t embedding_dim,
    int64_t numel,
    torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> partial_segment_offset,
    int64_t num_of_partial_segments,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> grad_weight_per_partial
) 
{
    const int partial_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (partial_id >= num_of_partial_segments) {
        return;
    }
    const int idx_begin = partial_segment_offset[partial_id];
    const int idx_end = (partial_id == num_of_partial_segments - 1) ? numel : partial_segment_offset[partial_id + 1];

    scalar_t grad_acc = 0;
    for (int idx = idx_begin; idx < idx_end; ++idx) {
        const int orig_hash_idx = orig_hash_idx_idx[idx];    // orig_idx in range [0, |indices| x embedding_dim)
        const int orig_cat_idx = orig_hash_idx / embedding_dim; // in range [0, |indices|)
        const int feature_idx =  orig_hash_idx % embedding_dim;     // in range [0, embedding_dim)
        const int bag_idx = offset2bag[orig_cat_idx];     
        grad_acc += output_grad[bag_idx][feature_idx]; 
    }
    grad_weight_per_partial[partial_id] = grad_acc;

}

template<typename scalar_t>
__global__ void sum_and_scatter(
    torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> sorted_unique_weight_idx,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> grad_weight_per_segment,
    torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> partical_per_segment_offset,
    int64_t num_segments,
    int64_t num_of_partial_segments,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> weight_grad
)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_segments) {
        return;
    }
    const int weight_idx = sorted_unique_weight_idx[gid];

    const int idx_begin = partical_per_segment_offset[gid];
    const int idx_end = (gid == num_segments - 1) ? num_of_partial_segments : partical_per_segment_offset[gid + 1];
    scalar_t grad_acc = 0;
    for (int idx = idx_begin; idx < idx_end; ++idx) {
        grad_acc += grad_weight_per_segment[idx];
    }
    weight_grad[weight_idx] = grad_acc;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> hashed_embedding_bag_cuda_forward(
    const torch::Tensor& hashed_weights,
    const torch::Tensor& indices,
    const torch::Tensor& offsets,
    const int64_t mode,
    const int64_t embedding_dim)
{
    int64_t numIndices = indices.size(0);
    int64_t numBags = offsets.size(0);
    int64_t hashedWeightSize = hashed_weights.size(0);
    auto bag_size = at::empty(offsets.sizes(), indices.options());
    auto offset2bag = 
        at::empty({indices.size(0)}, indices.options());
    auto hashed_index = at::empty({indices.size(0), embedding_dim}, indices.options());
    auto output = at::empty({numBags, embedding_dim}, hashed_weights.options());
    torch::Tensor max_indices;
    if (mode == MODE_MAX) {
        max_indices = at::empty({numBags, embedding_dim}, indices.options());
    } else {
        max_indices = at::empty({0, 0}, indices.options());
    } 
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

#ifdef __HIP_PLATFORM_HCC__
    dim3 block = dim3(64, 4);
#else
    dim3 block = dim3(32, 8);
#endif
    int grid = 1024;

    AT_DISPATCH_FLOATING_TYPES(hashed_weights.type(), "hashed_embedding_bag_cuda", ([&] {
        hashed_embedding_bag_update_output_kernel<scalar_t><<<grid, block, 0, stream>>>(
            indices.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            offsets.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            hashed_weights.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            offset2bag.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            numIndices,
            numBags,
            embedding_dim,
            hashedWeightSize,
            mode,
            hashed_index.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
            bag_size.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            max_indices.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>());
    }));

    return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>(
        output, offset2bag, bag_size, max_indices, hashed_index);
}

torch::Tensor hashed_embedding_bag_sum_backward(
    const torch::Tensor& output_grad,
    const torch::Tensor& indices,
    const torch::Tensor& offsets,
    const torch::Tensor& offset2bag,
    const torch::Tensor& hash_index,

    int64_t num_weights,
    int64_t embedding_dim)
{
    int64_t numIndices = indices.size(0);
    int64_t numBags = offsets.size(0);
    torch::Tensor weight_grad = torch::zeros({num_weights}, output_grad.options());

    if (numIndices == 0) {
        return weight_grad;
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    torch::Tensor flattened_hash_index = hash_index.flatten();
    int64_t numel = flattened_hash_index.size(0);

    // hash_index is a |indices| x embedding_dim Tensor, contains the index in hashed weight for each input indices x embedding dim.
    // the hash_index is flattened, and then we want to sort it, we use orig_hash_idx_idx to keep track of its orignal indices.
    auto sorted_hash_idx = at::empty_like(flattened_hash_index, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    auto orig_hash_idx_idx = at::empty_like(flattened_hash_index, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    using device_ptr = thrust::device_ptr<int64_t>;
    {
        sorted_hash_idx.copy_(flattened_hash_index);


        auto count_iter = thrust::counting_iterator<int64_t>(0);
        auto orig_hash_idx_idx_data = device_ptr(orig_hash_idx_idx.data_ptr<int64_t>());
        thrust::copy(count_iter, count_iter + numel, orig_hash_idx_idx_data);

        auto sorted_hash_idx_data = device_ptr(sorted_hash_idx.data_ptr<int64_t>());
        thrust::sort_by_key(
            sorted_hash_idx_data, 
            sorted_hash_idx_data + numel, 
            orig_hash_idx_idx_data);
    }

    // There may be many duplicates in the hash_index, now it's sorted, we find the start index for each hash_index value.
    // then we can get the count for each hash_index_value.
    auto segment_offsets = at::empty({numel}, orig_hash_idx_idx.options());
    auto sorted_unique_weight_idx = at::empty_like(sorted_hash_idx, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    int64_t num_segments;
    {
        auto sorted_hash_idx_data = device_ptr(sorted_hash_idx.data_ptr<int64_t>());
        auto sorted_unique_weight_idx_data = device_ptr(sorted_unique_weight_idx.data_ptr<int64_t>());
        auto iter_end_pair = thrust::unique_by_key_copy(
            sorted_hash_idx_data,
            sorted_hash_idx_data + numel,
            thrust::make_counting_iterator(0),
            sorted_unique_weight_idx_data,
            thrust::device_ptr<int64_t>(segment_offsets.data_ptr<int64_t>())
        );
        num_segments = thrust::get<0>(iter_end_pair) - sorted_unique_weight_idx_data;
    }

    // We split the segments up into sizes of `NROWS_PER_THREAD`
    // Compute the number partial-segments per segment (some partial-segments 
    // may not be the full `NROWS_PER_THREAD` number of rows)
    auto partials_per_segment = at::empty({num_segments}, orig_hash_idx_idx.options());
    {
        krn_partials_per_segment<<<ceil_div(num_segments, 32), 32, 0, stream>>> (
            partials_per_segment.data_ptr<int64_t>(),
            segment_offsets.data_ptr<int64_t>(),
            num_segments,
            numel);
    }


    // In order to compute `partial_segment_offset`, which is the start index
    // of each partial-segment in `sorted_indices`, we need to compute the
    // start position of each _segment_ in `partial_segment_offset`.
    // Unit: index in `partial_segment_offset`
    auto partials_per_segment_offset = at::empty({num_segments}, orig_hash_idx_idx.options());
    thrust::exclusive_scan(
        device_ptr(partials_per_segment.data_ptr<int64_t>()),
        device_ptr(partials_per_segment.data_ptr<int64_t>() + num_segments),
        device_ptr(partials_per_segment_offset.data_ptr<int64_t>())
    );

    // The total number of partial-segments is the sum of `partials_per_segment_offset`
    const int num_of_partial_segments = partials_per_segment[num_segments - 1].item<int64_t>() +
        partials_per_segment_offset[num_segments - 1].item<int64_t>();

    // Now we can compute the start position of each partial-segment
    // Unit: index in `sorted_indices` and `orig_indices`
    auto partial_segment_offset = at::empty({num_of_partial_segments}, orig_hash_idx_idx.options());
    {
        krn_partial_segment_offset<<<ceil_div(num_segments, 32), 32, 0, stream>>> (
            partial_segment_offset.data_ptr<int64_t>(),
            partials_per_segment.data_ptr<int64_t>(),
            partials_per_segment_offset.data_ptr<int64_t>(),
            segment_offsets.data_ptr<int64_t>(),
            num_segments);
    }
    auto grad_weight_per_segment = at::empty({num_of_partial_segments}, weight_grad.options());

    const int block = NWEIGHT_PER_THREAD;
    const int grid = ceil_div(num_of_partial_segments, block);
    AT_DISPATCH_ALL_TYPES(weight_grad.scalar_type(), "hashed_embedding_bag_backward_cuda", ([&] {
        compute_grad_weight_bags<scalar_t><<<grid, block, 0, stream>>>(
            orig_hash_idx_idx.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            output_grad.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            offset2bag.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            embedding_dim,
            numel,
            partial_segment_offset.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            num_of_partial_segments,
            grad_weight_per_segment.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>()
        );
        const int grid2 = ceil_div(num_segments, block);
        sum_and_scatter<scalar_t><<<grid2, block, 0, stream>>>(
            sorted_unique_weight_idx.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            grad_weight_per_segment.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            partials_per_segment_offset.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            num_segments,
            num_of_partial_segments,
            weight_grad.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>()
        );
    }));
    

    return weight_grad;
}

torch::Tensor hashed_embedding_bag_cuda_backward(
    const torch::Tensor& grad_,
    const torch::Tensor& indices,
    const torch::Tensor& offsets,
    const torch::Tensor& offset2bag,
    const torch::Tensor& bag_size_,
    const torch::Tensor& max_indices_,
    const torch::Tensor& hashed_index,
    int64_t num_weights,
    bool scale_grad_by_freq,
    int64_t mode,
    int64_t embedding_dim)
{
    torch::Tensor grad = grad_.contiguous();
    switch (mode) {
        case MODE_SUM:
            return hashed_embedding_bag_sum_backward(
                grad_,
                indices,
                offsets,
                offset2bag,
                hashed_index,
                num_weights,
                embedding_dim);
        case MODE_MEAN:
        case MODE_MAX:
            //return hashed_embedding_bag_cuda_max()
        default:
            return torch::Tensor();
    }
}

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> hashed_embedding_bag_forward(
    const torch::Tensor& hashed_weights,
    const torch::Tensor& indices,
    const torch::Tensor& offsets,
    //const bool scale_grad_by_freq,
    const int64_t mode,
    const int64_t embedding_dim) 
{

    CHECK_INPUT(hashed_weights);
    CHECK_INPUT(indices);
    CHECK_INPUT(offsets);

    return hashed_embedding_bag_cuda_forward(hashed_weights, indices, offsets, mode, embedding_dim);
}


torch::Tensor hashed_embedding_bag_backward(
    const torch::Tensor& grad,
    const torch::Tensor& indices,
    const torch::Tensor& offsets,
    const torch::Tensor& offset2bag,
    const torch::Tensor& bag_size_,
    const torch::Tensor& max_indices_,
    const torch::Tensor& hashed_index_,
    int64_t num_weights,
    bool scale_grad_by_freq,
    int64_t mode,
    int64_t embedding_dim)
{
    CHECK_CUDA(grad);
    CHECK_INPUT(indices);
    CHECK_INPUT(offsets);
    CHECK_INPUT(offset2bag);
    CHECK_INPUT(bag_size_);
    CHECK_INPUT(max_indices_);
    return hashed_embedding_bag_cuda_backward(
        grad,
        indices,
        offsets,
        offset2bag,
        bag_size_,
        max_indices_,
        hashed_index_,
        num_weights,
        scale_grad_by_freq,
        mode,
        embedding_dim
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &hashed_embedding_bag_forward, "hash embedding forward (CUDA)");
  m.def("backward", &hashed_embedding_bag_backward, "hash embedding backward (CUDA)");
  m.def("hash", &hash_func, "hash function");
}