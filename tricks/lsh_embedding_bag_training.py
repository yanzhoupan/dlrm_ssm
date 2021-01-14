from typing import Optional
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import init
import torch.nn.functional as F
import math
import pickle
from min_hash_generator import SparseBitVectorMinHashGenerator
import numpy as np


def make_offset2bag(offsets, indices):
    offsets2bag = torch.zeros(indices.size(0) + 1, dtype=indices.dtype, device=offsets.device)
    offsets2bag.index_add_(0, offsets, torch.ones_like(offsets, memory_format=torch.legacy_contiguous_format))
    offsets2bag[0] -= 1
    offsets2bag = offsets2bag.cumsum(0)
    offsets2bag.resize_(indices.size(0))
    return offsets2bag


class LshEmbeddingBigBag(nn.Module):

    def __init__(self, val_indices, _weight, mode="sum", val_idx_offset=0):
        """
        Create a LSH embedding bag layer with val_indices(generate from partial data), allowing different categories to share weights.

        :param val_indices: a dictionary contains the values and the corresponding indices. (can be preprocessed as stored)
        :param _weight: the shared weight for LSH embedding
        :param mode: "sum" or "mean", the way to aggregate embedded values in each bag.
        :param val_idx_offset: the index offset for each value.
        """

        super(LshEmbeddingBigBag, self).__init__()

        self.val_indices = val_indices
        self.hashed_weight = _weight
        self.lsh_weight_size = self.hashed_weight.numel()
        self.val_idx_offset = val_idx_offset
        # print("weight(embedding table): ", self.hashed_weight)
        assert (mode in ["sum", "mean"])
        self._mode = mode
        print("LSH embedding bag, weight id: ", id(self.hashed_weight))

    def forward(self,
                indices: torch.LongTensor,
                offsets: Optional[torch.LongTensor] = None,
                per_index_weights: Optional[torch.Tensor] = None):
        """
        Forward process to the embedding bag layer.
        :param indices: Tensor containing bags of indices into the embedding matrix.
        :param offsets: Only used when indices is 1D. offsets determines the starting index position of each bag
        (sequence)in input.
        :param per_index_weights: a tensor of float / double weights, or None to indicate all weights should be taken to
        be 1. If specified, per_sample_weights must have exactly the same shape as input and is treated as having the
        same offsets, if those are not None.
        :return: an #bag x embedding_dim Tensor.
        """

        # always move indices to cpu, as we need to get its corresponding minhash values from table in memory
        # indices = indices.cpu()
        indices += self.val_idx_offset

        # Check input validation.
        if per_index_weights is not None and indices.size() != per_index_weights.size():
            raise ValueError("embedding_bag: If per_index_weights ({}) is not None, "
                             "then it must have the same shape as the indices ({})"
                             .format(per_index_weights.shape, indices.shape))
        if indices.dim() == 2:
            if offsets is not None:
                raise ValueError("if input is 2D, then offsets has to be None"
                                 ", as input is treated is a mini-batch of"
                                 " fixed length sequences. However, found "
                                 "offsets of type {}".format(type(offsets)))
            offsets = torch.arange(0, indices.numel(), indices.size(1), dtype=torch.long, device=indices.device)
            indices = indices.reshape(-1)
            if per_index_weights is not None:
                per_sample_weights = per_index_weights.reshape(-1)
        elif indices.dim() == 1:
            if offsets is None:
                raise ValueError("offsets has to be a 1D Tensor but got None")
            if offsets.dim() != 1:
                raise ValueError("offsets has to be a 1D Tensor")
        else:
            ValueError("input has to be 1D or 2D Tensor,"
                       " but got Tensor of dimension {}".format(input.dim()))

        num_bags = offsets.size(0)

        
        # use partial data to calculate the lsh_weight_index on the fly:
        lsh_weight_index = []
        input_size = 128354
        embedding_dim = 3

        min_hash_gen = SparseBitVectorMinHashGenerator(input_size, embedding_dim, 2)

        for val_id in indices:
            lsh_weight_index.append(min_hash_gen.generate(self.val_indices[int(val_id)]))

        lsh_weight_index = torch.from_numpy(np.stack(lsh_weight_index, axis=0))
        print("forward:", lsh_weight_index)
        # get the min-hash for each category value, note that lsh_weight_index is in cpu memory
        # lsh_weight_index = self._minhash_table[indices] % self.lsh_weight_size

        # move the min-hash values to target device
        # lsh_weight_index = lsh_weight_index.to(self.hashed_weight.device)
        lsh_weight_index %= self.lsh_weight_size

        # indices_embedding_vector is a |indices| x |embedding_dim| tensor.
        indices_embedding_vectors = self.hashed_weight[lsh_weight_index]

        # multiply embedding vectors by weights
        if per_index_weights is not None:
            # per_index_weights = per_index_weights.to(indices_embedding_vectors.device)
            indices_embedding_vectors *= per_index_weights[:, None]

        offsets2bag = make_offset2bag(offsets, indices)

        if self._mode == "sum" or self._mode == "mean":
            result = \
                torch.zeros(num_bags, embedding_dim, dtype=indices_embedding_vectors.dtype,
                            device=self.hashed_weight.device)
            result.index_add_(0, offsets2bag, indices_embedding_vectors)
            if self._mode == "sum":
                return result

            # self._mode == "mean":
            bag_size = make_bag_size(offsets, indices).to(result.device)
            result /= bag_size[:, None]
            return result



def generatePartialDataValIndices():
    data = np.load('./input/kaggleAdDisplayChallenge_processed.npz')
    data_num, cat_num = data["X_cat"].shape # (45840617, 26)
    ratio = 0.0028 # using 125k samples
    partial_idx = np.random.choice(np.arange(data_num), size=int(data_num * ratio), replace=False)
    partial_cat_data = data['X_cat'][partial_idx]

    np.savez(r'./input/cat_counts.npz', cat_counts = data['counts'])

    base = 0
    val_indices = defaultdict(lambda:[])
    # generate signiture matrix for category values (partial data)
    for fea_id in tqdm(range(cat_num)):
        cat_fea = partial_cat_data[:, fea_id]
        
        for doc_id in range(len(cat_fea)): # loop over docs
            val_indices[cat_fea[doc_id] + base].append(doc_id)
            
        for val in range(data['counts'][fea_id]):
            if val_indices[val+base] == []: 
                val_indices[val+base] = [45840618] # set val_indices to a fixed place if never seen it
        base += data['counts'][fea_id]

    with open("./input/val_indices_125k.pkl", 'wb') as f:
        pickle.dump(dict(val_indices), f, pickle.HIGHEST_PROTOCOL)


def lshEmbeddingBagTest():
    val_indices = {0:[1,2,3], 1:[2,3,4], 2:[3]}
    hashed_weight = nn.Parameter(torch.from_numpy(np.random.uniform(
                        low=-np.sqrt(1 / 2), high=np.sqrt(1 / 2), size=((int(sum([1,1,1]) * 8 * 0.5),))
                ).astype(np.float32)))

    print("hashed_weight", hashed_weight)

    embedding_bag = LshEmbeddingBigBag(val_indices, hashed_weight, mode="sum", val_idx_offset=0)
    test_indices = torch.LongTensor([0, 1, 2])

    test_offset = torch.LongTensor([0, 1, 2])
    # test_per_sample_weight = torch.DoubleTensor([1, 2, 1])
    print(embedding_bag.forward(test_indices, test_offset))

    embedding_bag = embedding_bag.cuda()
    test_indices = torch.LongTensor([0, 1, 2, 3, 4]).cuda()

    test_offset = torch.LongTensor([0, 2]).cuda()
    test_per_sample_weight = torch.DoubleTensor([2, 3, 2, 3, 2]).cuda()



if __name__ == "__main__":
    lshEmbeddingBagTest()