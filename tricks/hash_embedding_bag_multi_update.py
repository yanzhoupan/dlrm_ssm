from typing import Optional
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import init
import torch.nn.functional as F
import xxhash
import math
# from qr_embedding_bag import QREmbeddingBag
import numpy as np

def getMaxNumOfUniqueValue():
    max_len = 0
    for idx in range(26):
        fea_dict = np.load('./input/train_fea_dict_'+str(idx)+'.npz')
        max_len = max(0, len(fea_dict["unique"]))
    return max_len

HASHED_WEIGHT = torch.Tensor(0)

class HashEmbeddingBagMultiUpdate(nn.Module):

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 compression: float,
                 update_count=1,
                 lens=(),
                 hash_seed=2,
                 mode="sum",
                 sparse=False,
                 _weight: Optional[torch.Tensor] = None
                 ):
        super(HashEmbeddingBagMultiUpdate, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.compression = compression
        self.lens = lens
        self.hash_seed = hash_seed
        self.mode = mode
        self.sparse = sparse
        self.xxhash = xxhash
        self.update_count = update_count

        if _weight is None:
            self.hashed_weight_size = int(self.num_embeddings * self.embedding_dim * compression)
            # self.hashed_weight_size = max(self.hashed_weight_size, 16)
            self.hashed_weight = Parameter(torch.Tensor(self.hashed_weight_size))
            W = np.random.uniform(
                        low=-np.sqrt(1 / self.num_embeddings), high=np.sqrt(1 / self.num_embeddings), size=((int(self.num_embeddings * self.embedding_dim * self.compression), ))
                    ).astype(np.float32)
            self.hashed_weight.data = torch.tensor(W, requires_grad=True)
        else:
            # print('_weight passed in!')
            #assert len(_weight.shape) == 1 and _weight.shape[0] == weight_size, \
            #    'Shape of weight does not match num_embeddings and embedding_dim'
            self.hashed_weight = _weight
            self.hashed_weight_size = self.hashed_weight.numel()

        # if not self.lens: # use a hashed weight vector for each hash table
        #     self.hashed_weight_size = int(self.num_embeddings * self.embedding_dim * compression)
        #     # self.hashed_weight_size = max(self.hashed_weight_size, 16)
        #     self.hashed_weight = Parameter(torch.Tensor(self.hashed_weight_size))

        # else: # use a shared weight vector for all the hash tables
        #     self.hashed_weight_size = int(sum(self.lens) * self.embedding_dim * compression)
        #     global HASHED_WEIGHT
        #     if HASHED_WEIGHT.size() == torch.Size([0]):
        #         print("Using a shared weight vector for all the hash tables, hashed weight size:: ", self.hashed_weight_size)
        #         HASHED_WEIGHT = torch.Tensor(self.hashed_weight_size)
        #         W = np.random.uniform(
        #             low=-np.sqrt(1 / sum(self.lens)), high=np.sqrt(1 / sum(self.lens)), size=((self.hashed_weight_size, ))
        #         ).astype(np.float32)
        #         HASHED_WEIGHT.data = torch.tensor(W, requires_grad=True)
        #     self.hashed_weight = Parameter(HASHED_WEIGHT)
        
        self.weight_idx_list = []
        for idx in range(self.update_count):
            self.weight_idx_list.append(self.xxhash_func(self.hashed_weight_size, self.num_embeddings, self.embedding_dim, "idxW"+str(idx)))


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
        idx = torch.LongTensor(size_out, size_in)
        for i in range(size_out):
            for j in range(size_in):
                # idx[i, j] = (i * 9824516537 + j) % hN
                idx[i, j] = ((i * 32452843 + j * 86028121) % 512927357) % hN
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
        res = F.embedding_bag(x, self.hashed_weight[self.weight_idx_list[0]], offsets=offsets, mode=self.mode, sparse=self.sparse)
        for idx in range(1, self.update_count):
            res += F.embedding_bag(x, self.hashed_weight[self.weight_idx_list[idx]], offsets=offsets, mode=self.mode, sparse=self.sparse)
        return res




def HashEmbeddingBagMultiUpdateTest():
    # test HashEmbeddingBagMultiUpdate
    embedding_bag = HashEmbeddingBagMultiUpdate(10, 5, 1.0)
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
    HashEmbeddingBagMultiUpdateTest()
