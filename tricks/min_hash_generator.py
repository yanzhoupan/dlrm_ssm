import torch
import numpy as np
from typing import Set, Any
from primesieve import nth_prime
import xxhash

# minhash generator (based on permutation)
class SparseBitVectorMinHashGenerator:
    def __init__(self,
                 input_size,
                 num_perm=128,
                 num_minhash=1):
        self.input_size = input_size
        self.num_perm = num_perm
        self.num_minhash = num_minhash
        # self.permutations = [np.random.permutation(self.input_size) for _ in range(self.num_perm)]
        # self.permutation_start = np.random.randint(0, self.input_size, self.num_perm)
        
        self.permutation_hashes = []
        for _ in range(self.num_minhash):
            self.permutation_hashes.append([self.generate_hash() for _ in range(self.num_perm)])

        # self.permutation_hashes = [self.generate_hash() for _ in range(self.num_perm)]
        # self.permutation_hashes_exp = [self.generate_hash() for _ in range(self.num_perm)]

    def generate_hash(self):
        c = nth_prime(1000000)
        a_idx = np.random.randint(10000, 1000000)
        a = nth_prime(a_idx)
        b_idx = np.random.randint(10000, 1000000)
        b = nth_prime(b_idx)
        # c_idx = np.random.randint(10000, 1000000) # choose a large c, a and b is in range(1, c)
        # c = nth_prime(c_idx)
        return lambda x: (a * x + b) % c % self.input_size

    def generate_old(self, sparse_bit_vector):
        result = np.full(self.num_perm, self.input_size, dtype=np.int)
        for r in sparse_bit_vector:
            for i in range(self.num_perm):
                hashed_value = self.permutation_hashes[i](r)
                result[i] = min(result[i], hashed_value)
        
        for i in range(self.num_perm):
            key = '{}_{}'.format(i, 'minHash')
            xxhash.xxh32(key, 2).intdigest()
        return result

    
    def generate(self, sparse_bit_vector): # TODO: sparse_bit_vector becomes val_sp (torch sparse vector) and use sparse_matmul()
        result = np.full(self.num_perm, self.input_size, dtype=np.int)
        results = []
        for _ in range(self.num_minhash):
            results.append(np.full(self.num_perm, self.input_size, dtype=np.int))
        
        for r in sparse_bit_vector:
            for i in range(self.num_perm):
                for hash_idx in range(self.num_minhash):
                    hashed_value = self.permutation_hashes[hash_idx][i](r)
                    results[hash_idx][i] = min(results[hash_idx][i], hashed_value)
        
        for i in range(self.num_perm):
            key = ""
            for hash_idx in range(self.num_minhash):
                key += str(results[hash_idx][i]) + "_"
            key += "minhash"
            result[i] = xxhash.xxh32(key, 2).intdigest()
        return result


if __name__ == "__main__":
    gen = SparseBitVectorMinHashGenerator(1024,1024)

    i = torch.LongTensor([[0, 0], [0, 1], [1, 0], [1, 2], [2, 3]])
    v = torch.ones(5)
    x = torch.sparse.FloatTensor(i.t(), v, torch.Size([3, 1024]))

    line0 = torch.index_select(x, 0, torch.LongTensor([0]))
    print(line0)
    print(line0._indices()[:, 1].data.numpy())
    embedding0 = gen.generate(line0._indices()[:, 1].data.numpy())
    print(embedding0)

    line1 = torch.index_select(x, 0, torch.LongTensor([1]))
    print(line1)
    print(line1._indices()[:, 1].data.numpy())
    embedding1 = gen.generate(line1._indices()[:, 1].data.numpy())
    print(embedding1)

    # intersection = [i for i in embedding0 if i in embedding1]
    intersection = 0
    for idx in range(len(embedding0)):
        if embedding0[idx] == embedding1[idx]:
            intersection += 1
    print(intersection/len(embedding0))