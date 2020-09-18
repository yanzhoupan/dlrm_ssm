import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import init
import torch.nn.functional as F
import xxhash
import math


class HashEmbeddingBag(nn.Module):

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 compression: float,
                 hash_seed=2):
        super(HashEmbeddingBag, self).__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.compression = compression
        self.hash_seed = hash_seed

        self.hashed_weight_size = math.ceil(self.num_embeddings * self.embedding_dim * compression)
        self.xxhash = xxhash
        self.hashed_weight = Parameter(torch.Tensor(self.hashed_weight_size))
        self.weight_idx = self.hash_func(self.hashed_weight_size, self.num_embeddings, self.embedding_dim, "idxW")

    def hash_func(self, hN, size_out, size_in, extra_str=''):
        '''
        Hash matrix indices to an index in the compressed vector
        representation.

        Returns a matrix of indices with size size_out x size_in,
        where the indices are in the range [0,hN).
        '''
        idx = torch.LongTensor(size_out, size_in)
        for i in range(size_out):
            for j in range(size_in):
                key = '{}_{}{}'.format(i, j, extra_str)

                # Wrap hashed values to the compressed range
                idx[i, j] = self.xxhash.xxh32(key, self.hash_seed).intdigest() % hN

        return idx

    def forward(self, x):
        return F.embedding_bag(x, self.hashed_weight[self.weight_idx])

def hashEmbeddingBagTest():
    embedding_bag = HashEmbeddingBag(10, 5, 1.0/2.0)
    test_input = torch.randint(0, 10, torch.Size([3, 2]))
    print(embedding_bag.forward(test_input))

if __name__ == "__main__":
    hashEmbeddingBagTest()
