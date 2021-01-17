from typing import Optional
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import init
import torch.nn.functional as F
import xxhash
import math
import numpy as np

def getMaxNumOfUniqueValue():
    max_len = 0
    for idx in range(26):
        fea_dict = np.load('./input/train_fea_dict_'+str(idx)+'.npz')
        max_len = max(0, len(fea_dict["unique"]))
    return max_len

HASHED_WEIGHT = torch.Tensor(0)

class HashEmbeddingBag(nn.Module):

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 compression=1.0,
                 hash_seed=2,
                 mode="sum",
                 sparse=False,
                 _weight: Optional[torch.Tensor] = None
                 ):
        super(HashEmbeddingBag, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.compression = compression
        self.hash_seed = hash_seed
        self.mode = mode
        self.sparse = sparse
        self.xxhash = xxhash

        if _weight is None:
            self.hashed_weight_size = int(self.num_embeddings * self.embedding_dim * compression)
            # self.hashed_weight_size = max(self.hashed_weight_size, 16)
            self.hashed_weight = Parameter(torch.Tensor(self.hashed_weight_size))
            W = np.random.uniform(
                        low=-np.sqrt(1 / self.num_embeddings), high=np.sqrt(1 / self.num_embeddings), size=((int(self.num_embeddings * self.embedding_dim * self.compression), ))
                    ).astype(np.float32)
            self.hashed_weight.data = torch.tensor(W, requires_grad=True)
        else:
            self.hashed_weight = _weight
            self.hashed_weight_size = self.hashed_weight.numel()
        
        self.weight_idx = self.uni_hash_func(self.hashed_weight_size, self.num_embeddings, self.embedding_dim, "idxW")
        # self.weight_idx = torch.LongTensor(self.num_embeddings, self.embedding_dim).random_(0, self.hashed_weight_size)


    def xxhash_func(self, hN, size_out, size_in, extra_str=''):
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
    
    def uni_hash_func(self, hN, size_out, size_in, extra_str=''):
        '''
        This is a determinestic hash function
        '''

        # a, b, c, d= 32452843, 86028121, 15485863, 512927357
        # mat1 = torch.arange(0, size_out, out=torch.LongTensor()).repeat(size_in, 1).t()
        # mat2 = torch.arange(0, size_in, out=torch.LongTensor()).repeat(size_out, 1)
        # idx_mat = (( mat1 * a + mat2 * b + c) % d) % hN
        # return idx_mat

        idx = torch.LongTensor(size_out, size_in)
        for i in range(size_out):
            for j in range(size_in):
                idx[i, j] = ((i * 32452843 + j * 86028121 + 15485863) % 512927357) % hN
        return idx

    
    def cantor_pairing_hash_func(self, hN, size_out, size_in, extra_str=''):
        '''
        Cantor pairing hash function (determinestic)
        '''
        idx = torch.LongTensor(size_out, size_in)
        for i in range(size_out):
            for j in range(size_in):
                idx[i, j] = ((i + j) * (i + j + 1) // 2 + j) % hN
        return idx
    
    def linear_hash_func(self, hN, size_out, size_in, extra_str=''):
        '''
        This is a linear hash function that map a 2D pair to a 1D vector (determinestic)
        '''
        print("Using linear hash...")
        idx = torch.LongTensor(size_out, size_in)
        for i in range(size_out):
            for j in range(size_in):
                idx[i, j] = (i * size_in + j) % hN
        return idx

    def forward(self, x, offsets=None):
        # self.weight_idx = self.weight_idx.to(x.device)
        # self.hashed_weight = self.hashed_weight.to(x.device)
        # print("Forward: ", self.hashed_weight, self.hashed_weight[self.weight_idx])
        return F.embedding_bag(x, self.hashed_weight[self.weight_idx], offsets=offsets, mode=self.mode, sparse=self.sparse)

# class SharedWeightHashEmbeddingBags(nn.Module):

def hashEmbeddingBagTest():
    # test hashEmbeddingBag
    embedding_bag = HashEmbeddingBag(10, 5, 1.0)
    # test_input = torch.randint(0, 10, torch.Size([5,]))
    # print("Test input", test_input)
    print("Embedding weight before forward: ", embedding_bag.hashed_weight)
    print("Result:", embedding_bag(torch.tensor([0,0,1,1,2]), torch.tensor([0,1,2,3,4])))
    print("Embedding weight after forward: ", embedding_bag.hashed_weight)

    # the original EmbeddingBag
    n, m = 10, 5
    emb = nn.EmbeddingBag(n, m, mode="sum", sparse=True)
    emb.weight.data = embedding_bag.hashed_weight.data.reshape(n,m)
    # initialize embeddings
    # nn.init.uniform_(EE.weight, a=-np.sqrt(1 / n), b=np.sqrt(1 / n))
    # W = np.random.uniform(
        # low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
    # ).astype(np.float32)
    # approach 1
    # emb.weight.data = torch.tensor(W, requires_grad=True)
    print("Original emb: ", emb.weight.data, emb(torch.tensor([0,0,1,1,2]), torch.tensor([0,1,2,3,4])))

    target = [0,0,0,0,1]



if __name__ == "__main__":
    hashEmbeddingBagTest()
