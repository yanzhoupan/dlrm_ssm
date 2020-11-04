from typing import Optional
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import init
import torch.nn.functional as F
import math


def make_bag_size(offsets, indices):
    bag_size = torch.zeros(offsets.size(), dtype=indices.dtype, device=offsets.device)
    bag_size[:-1] = offsets[1:] - offsets[:-1]
    bag_size[-1] = indices.size(0) - offsets[-1]
    return bag_size


def make_offset2bag(offsets, indices):
    offsets2bag = torch.zeros(indices.size(0) + 1, dtype=indices.dtype, device=offsets.device)
    offsets2bag.index_add_(0, offsets, torch.ones_like(offsets, memory_format=torch.legacy_contiguous_format))
    offsets2bag[0] -= 1
    offsets2bag = offsets2bag.cumsum(0)
    offsets2bag.resize_(indices.size(0))
    return offsets2bag


class LshEmbeddingBag(nn.Module):

    def __init__(self, minhash_table: torch.LongTensor, compression=1.0, mode="sum"):
        """
        Create a LSH embedding bag layer with a min-hashing table.
        The min-hashing table contains known min-hash values for each category value.
        The table is a N x D LongTensor, in which N is the number of category values, D is the embedding dimension.
        The kth line of the table is the min-hash value for the kth category value.
        :param minhash_table: the min-hash table, which should be a N x D LongTensor. The table is forced to be stored
        in CPU memory.
        :param compression: the compression rate compare to native embedding function. native embedding function has a
        N x D weight table in GPU memory. The LshEmbeddingBag only uses compression x N x D GPU memory.
        :param mode: "sum" or "mean", the way to aggregate embedded values in each bag.
        """

        super(LshEmbeddingBag, self).__init__()

        self._minhash_table = minhash_table
        self._minhash_table = self._minhash_table.cpu().detach()
        num_embeddings = self._minhash_table.size(0)
        self.embedding_dim = self._minhash_table.size(1)

        self.lsh_weight_size = math.ceil(num_embeddings * self.embedding_dim * compression)
        self.hashed_weight = Parameter(torch.Tensor(self.lsh_weight_size))
        # print("weight(embedding table): ", self.hashed_weight)
        assert (mode in ["sum", "mean"])
        self._mode = mode

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
        indices = indices.cpu()

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

        # get the min-hash for each category value, note that lsh_weight_index is in cpu memory
        lsh_weight_index = self._minhash_table[indices]
        # print("In forward: ", lsh_weight_index, indices, self._minhash_table[indices], self.lsh_weight_size)

        # move the min-hash values to target device
        lsh_weight_index = lsh_weight_index.to(self.hashed_weight.device)
        lsh_weight_index %= self.lsh_weight_size

        # indices_embedding_vector is a |indices| x |embedding_dim| tensor.
        indices_embedding_vectors = self.hashed_weight[lsh_weight_index]
        # print('indices_embedding_vectors: ', lsh_weight_index, indices_embedding_vectors)

        # multiply embedding vectors by weights
        if per_index_weights is not None:
            per_index_weights = per_index_weights.to(indices_embedding_vectors.device)
            indices_embedding_vectors *= per_index_weights[:, None]
        # print("per_index_weights",per_index_weights)
        offsets2bag = make_offset2bag(offsets, indices)
        # print("offsets2bag: ", offsets2bag)
        if self._mode == "sum" or self._mode == "mean":
            result = \
                torch.zeros(num_bags, self.embedding_dim, dtype=indices_embedding_vectors.dtype,
                            device=self.hashed_weight.device)
            result.index_add_(0, offsets2bag, indices_embedding_vectors)
            if self._mode == "sum":
                return result

            # self._mode == "mean":
            bag_size = make_bag_size(offsets, indices).to(result.device)
            result /= bag_size[:, None]
            return result


def lshEmbeddingBagTest():
    min_hash_table = torch.LongTensor(
        [[0, 1],
         [2, 3],
         [0, 1],
         [6, 7],
         [8, 9]]
    )
    embedding_bag = LshEmbeddingBag(min_hash_table, .5, mode="mean")
    test_indices = torch.LongTensor([0, 1, 2, 3, 4])

    test_offset = torch.LongTensor([0,1,2,3,4])
    test_per_sample_weight = torch.DoubleTensor([2, 3, 2, 3, 2])
    print(embedding_bag.forward(test_indices, test_offset, test_per_sample_weight))

    embedding_bag = embedding_bag.cuda()
    test_indices = torch.LongTensor([0, 1, 2, 3, 4]).cuda()

    test_offset = torch.LongTensor([0, 2]).cuda()
    test_per_sample_weight = torch.DoubleTensor([2, 3, 2, 3, 2]).cuda()
    # print(embedding_bag.forward(test_indices, test_offset, test_per_sample_weight))

class LshEmbeddingBigBag(nn.Module):

    def __init__(self, minhash_table: torch.LongTensor, compression=1.0, mode="sum", _weight=None, val_idx_offset=0):
        """
        Create a LSH embedding bag layer with a big min-hashing table, allowing different categories to share weights.
        The min-hashing table contains known min-hash values for each category value.
        The table is a N x D LongTensor, in which N is the number of category values, D is the embedding dimension.
        The kth line of the table is the min-hash value for the kth category value.
        :param minhash_table: the min-hash table, which should be a N x D LongTensor. The table is forced to be stored
        in CPU memory.
        :param compression: the compression rate compare to native embedding function. native embedding function has a
        N x D weight table in GPU memory. The LshEmbeddingBag only uses compression x N x D GPU memory.
        :param mode: "sum" or "mean", the way to aggregate embedded values in each bag.
        """

        super(LshEmbeddingBigBag, self).__init__()

        self._minhash_table = minhash_table
        self._minhash_table = self._minhash_table.cpu().detach()
        num_embeddings = self._minhash_table.size(0)
        self.embedding_dim = self._minhash_table.size(1)

        # self.lsh_weight_size = math.ceil(num_embeddings * self.embedding_dim * compression)
        # self.hashed_weight = Parameter(torch.Tensor(self.lsh_weight_size))
        self.hashed_weight = _weight
        self.lsh_weight_size = self.hashed_weight.numel()
        self.val_idx_offset = val_idx_offset
        # print("weight(embedding table): ", self.hashed_weight)
        assert (mode in ["sum", "mean"])
        self._mode = mode

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
        indices = indices.cpu()
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

        # get the min-hash for each category value, note that lsh_weight_index is in cpu memory
        lsh_weight_index = self._minhash_table[indices]
        # print("In forward: ", lsh_weight_index, indices, self._minhash_table[indices], self.lsh_weight_size)

        # move the min-hash values to target device
        lsh_weight_index = lsh_weight_index.to(self.hashed_weight.device)
        lsh_weight_index %= self.lsh_weight_size

        # indices_embedding_vector is a |indices| x |embedding_dim| tensor.
        indices_embedding_vectors = self.hashed_weight[lsh_weight_index]
        # print('indices_embedding_vectors: ', lsh_weight_index, indices_embedding_vectors)

        # multiply embedding vectors by weights
        if per_index_weights is not None:
            per_index_weights = per_index_weights.to(indices_embedding_vectors.device)
            indices_embedding_vectors *= per_index_weights[:, None]
        # print("per_index_weights",per_index_weights)
        offsets2bag = make_offset2bag(offsets, indices)
        # print("offsets2bag: ", offsets2bag)
        if self._mode == "sum" or self._mode == "mean":
            result = \
                torch.zeros(num_bags, self.embedding_dim, dtype=indices_embedding_vectors.dtype,
                            device=self.hashed_weight.device)
            result.index_add_(0, offsets2bag, indices_embedding_vectors)
            if self._mode == "sum":
                return result

            # self._mode == "mean":
            bag_size = make_bag_size(offsets, indices).to(result.device)
            result /= bag_size[:, None]
            return result


if __name__ == "__main__":
    lshEmbeddingBagTest()
