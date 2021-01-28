# from typing import Optional
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

class HashVectorEmbeddingBag(nn.Module):

    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 compression=1.0,
                #  lens=(),
                 hash_seed=2,
                 mode="sum",
                 sparse=False,
                 _weight = None
                 ):
        super(HashVectorEmbeddingBag, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.compression = compression
        # self.lens = lens
        self.hash_seed = hash_seed
        self.mode = mode
        self.sparse = sparse
        self.xxhash = xxhash

        if _weight is None:
            self.hashed_weight_size = int(self.num_embeddings * compression)
            self.hashed_weight_size = max(self.hashed_weight_size, 1)
            self.hashed_weight = Parameter(torch.Tensor((self.hashed_weight_size, self.embedding_dim)))
            W = np.random.uniform(
                        low=-np.sqrt(1 / self.num_embeddings), high=np.sqrt(1 / self.num_embeddings), size=(self.hashed_weight_size, self.embedding_dim)
                    ).astype(np.float32)
            self.hashed_weight.data = torch.tensor(W, requires_grad=True)
        else:
            # print('_weight passed in!')
            #assert len(_weight.shape) == 1 and _weight.shape[0] == weight_size, \
            #    'Shape of weight does not match num_embeddings and embedding_dim'
            self.hashed_weight = _weight
            self.hashed_weight_size = self.hashed_weight.shape[0] # number of rows in the hashed table
        
        self.weight_idx = self.xxhash_func(self.hashed_weight_size, self.num_embeddings, "idxW")


    def xxhash_func(self, hN, size_out, extra_str=''):
        '''
        Hash matrix indices to an index in the compressed vector
        representation.

        Returns a vector of indices with size size_out,
        where the indices are in the range [0,hN).
        '''
        idx = torch.LongTensor(size_out)
        for i in range(size_out):
            key = '{}_{}'.format(i, extra_str)

            # Wrap hashed values to the compressed range
            idx[i] = self.xxhash.xxh32(key, self.hash_seed).intdigest() % hN
        return idx
    
    def uni_hash_func(self, hN, size_out, extra_str=''):
        '''
        This is a determinestic hash function
        '''
        idx = torch.LongTensor(size_out)
        for i in range(size_out):
            # idx[i, j] = (i * 9824516537 + j) % hN
            idx[i] = ((i * 32452843) % 512927357) % hN
        return idx
    
    def cantor_pairing_hash_func(self, hN, size_out, extra_str=''):
        '''
        Cantor pairing hash function (determinestic)
        '''
        idx = torch.LongTensor(size_out)
        for i in range(size_out):
            idx[i] = ((i ) * (i + 1) // 2 ) % hN
        return idx
    
    def linear_hash_func(self, hN, size_out, extra_str=''):
        '''
        This is a linear hash function that map a 2D pair to a 1D vector (determinestic)
        '''
        print("Using linear hash...")
        idx = torch.LongTensor(size_out)
        for i in range(size_out):
            idx[i] = (i * hN) % hN
        return idx

    def forward(self, x, offsets=None):
        # self.weight_idx = self.weight_idx.to(x.device)
        # self.hashed_weight = self.hashed_weight.to(x.device)
        # print("Forward: ", self.hashed_weight, self.hashed_weight[self.weight_idx])
        return F.embedding_bag(x, self.hashed_weight[self.weight_idx, :], offsets=offsets, mode=self.mode, sparse=self.sparse)


class MultiUpdateHashVectorEmbeddingBag(nn.Module):

    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 compression=1.0,
                 update_count=1,
                 hash_seed=2,
                 mode="sum",
                 sparse=False,
                 _weight = None
                 ):
        super(MultiUpdateHashVectorEmbeddingBag, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.compression = compression
        self.update_count = update_count
        # self.lens = lens
        self.hash_seed = hash_seed
        self.mode = mode
        self.sparse = sparse
        self.xxhash = xxhash

        if _weight is None:
            self.hashed_weight_size = int(self.num_embeddings * compression)
            self.hashed_weight_size = max(self.hashed_weight_size, 1)
            self.hashed_weight = Parameter(torch.Tensor((self.hashed_weight_size, self.embedding_dim)))
            W = np.random.uniform(
                        low=-np.sqrt(1 / self.num_embeddings), high=np.sqrt(1 / self.num_embeddings), size=(self.hashed_weight_size, self.embedding_dim)
                    ).astype(np.float32)
            self.hashed_weight.data = torch.tensor(W, requires_grad=True)
        else:
            # print('_weight passed in!')
            #assert len(_weight.shape) == 1 and _weight.shape[0] == weight_size, \
            #    'Shape of weight does not match num_embeddings and embedding_dim'
            self.hashed_weight = _weight
            self.hashed_weight_size = self.hashed_weight.shape[0] # number of rows in the hashed table
        
        self.weight_idx_list = []
        for idx in range(self.update_count):
            self.weight_idx_list.append(self.xxhash_func(self.hashed_weight_size, self.num_embeddings, "idxW"+str(idx)))


    def xxhash_func(self, hN, size_out, extra_str=''):
        '''
        Hash matrix indices to an index in the compressed vector
        representation.

        Returns a vector of indices with size size_out,
        where the indices are in the range [0,hN).
        '''
        idx = torch.LongTensor(size_out)
        for i in range(size_out):
            key = '{}_{}'.format(i, extra_str)

            # Wrap hashed values to the compressed range
            idx[i] = self.xxhash.xxh32(key, self.hash_seed).intdigest() % hN
        return idx
    
    def uni_hash_func(self, hN, size_out, extra_str=''):
        '''
        This is a determinestic hash function
        '''
        idx = torch.LongTensor(size_out)
        for i in range(size_out):
            # idx[i, j] = (i * 9824516537 + j) % hN
            idx[i] = ((i * 32452843) % 512927357) % hN
        return idx
    
    def cantor_pairing_hash_func(self, hN, size_out, extra_str=''):
        '''
        Cantor pairing hash function (determinestic)
        '''
        idx = torch.LongTensor(size_out)
        for i in range(size_out):
            idx[i] = ((i ) * (i + 1) // 2 ) % hN
        return idx
    
    def linear_hash_func(self, hN, size_out, extra_str=''):
        '''
        This is a linear hash function that map a 2D pair to a 1D vector (determinestic)
        '''
        print("Using linear hash...")
        idx = torch.LongTensor(size_out)
        for i in range(size_out):
            idx[i] = (i * size_in) % hN
        return idx

    def forward(self, x, offsets=None):
        # self.weight_idx = self.weight_idx.to(x.device)
        # self.hashed_weight = self.hashed_weight.to(x.device)
        # print("Forward: ", self.hashed_weight, self.hashed_weight[self.weight_idx])
        res = F.embedding_bag(x, self.hashed_weight[self.weight_idx_list[0], :], offsets=offsets, mode=self.mode, sparse=self.sparse)
        for idx in range(1, self.update_count):
            res += F.embedding_bag(x, self.hashed_weight[self.weight_idx_list[idx], :], offsets=offsets, mode=self.mode, sparse=self.sparse)
        return res

def hashEmbeddingBagTest():
    # test hashEmbeddingBag
    embedding_bag = HashVectorEmbeddingBag(10, 5, 1.0)
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
