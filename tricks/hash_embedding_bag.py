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

HASHED_WEIGHT = Parameter(torch.Tensor(0))

class HashEmbeddingBag(nn.Module):

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 compression: float,
                #  max_len=0,
                 lens=(),
                 hash_seed=2,
                 mode="sum"
                 ):
        super(HashEmbeddingBag, self).__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.compression = compression
        # self.max_len = max_len
        self.lens = lens
        self.hash_seed = hash_seed
        self.mode = mode

        self.xxhash = xxhash

        if not self.lens: # use a hashed weight vector for each hash table
            self.hashed_weight_size = math.ceil(self.num_embeddings * self.embedding_dim * compression)
            self.hashed_weight = Parameter(torch.Tensor(self.hashed_weight_size))
            torch.nn.init.normal(self.hashed_weight, mean=0, std=1)
        else: # use a shared weight vector for all the hash tables
            self.hashed_weight_size = math.ceil(sum(self.lens) * self.embedding_dim * compression)
            global HASHED_WEIGHT
            if HASHED_WEIGHT.size() == torch.Size([0]):
                HASHED_WEIGHT = Parameter(torch.Tensor(self.hashed_weight_size))
                torch.nn.init.normal(HASHED_WEIGHT, mean=0, std=1)
            self.hashed_weight = HASHED_WEIGHT

            # print("SharedHashbag, hashed_weight_size: ", len(hashed_weight))
            # self.hashed_weight = hashed_weight
        
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

    def forward(self, x, offsets=None):
        # if not self.lens:
        self.weight_idx = self.weight_idx.to(x.device)
        self.hashed_weight = self.hashed_weight.to(x.device)
        # print("Forward: ", x.device, self.hashed_weight.device, self.weight_idx.device)
        return F.embedding_bag(x, self.hashed_weight[self.weight_idx], offsets=offsets, mode=self.mode)
        # else:
        #     # global hashed_weight
        #     return F.embedding_bag(x, hashed_weight[self.weight_idx], offsets=offsets, mode=self.mode)



def hashEmbeddingBagTest():
    # test hashEmbeddingBag
    embedding_bag = HashEmbeddingBag(10, 5, 1.0/2.0, 3)
    test_input = torch.randint(0, 10, torch.Size([5,]))
    # print("Test input", test_input)
    print("Result:", embedding_bag.forward(torch.tensor([0,0,1,1,2]), torch.tensor([0,1,2,3,4])))

    # the original EmbeddingBag
    n, m = 10, 5
    emb = nn.EmbeddingBag(n, m, mode="sum", sparse=True)
    # initialize embeddings
    # nn.init.uniform_(EE.weight, a=-np.sqrt(1 / n), b=np.sqrt(1 / n))
    W = np.random.uniform(
        low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
    ).astype(np.float32)
    # approach 1
    emb.weight.data = torch.tensor(W, requires_grad=True)
    print("emb: ", emb.weight.data, emb(torch.tensor([0,0,1,1,2]), torch.tensor([0,1,2,3,4])))

if __name__ == "__main__":
    hashEmbeddingBagTest()
