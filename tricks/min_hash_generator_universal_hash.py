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


        self.A1s, self.A2s, self.A3s, self.Bases = [], [], [], []
        self.BIGPRIME = nth_prime(1000000)
        for _ in range(self.num_minhash):
            self.A1s.append(torch.randint(1, self.BIGPRIME, torch.Size((1, self.num_perm))))
            self.A2s.append(torch.randint(1, self.BIGPRIME, torch.Size((1, self.num_perm))))
            self.A3s.append(torch.randint(1, self.BIGPRIME, torch.Size((1, self.num_perm))))
            self.Bases.append(torch.randint(0, self.BIGPRIME, torch.Size((1, self.num_perm))))

        self.A4s = torch.randint(1, self.BIGPRIME, torch.Size((1, self.num_minhash)))
        self.B4 = np.random.randint(0, self.BIGPRIME)

    def generate(self, sparse_bit_vector):
        sparse_bit_vector = torch.LongTensor(sparse_bit_vector)
        sparse_bit_vector = sparse_bit_vector.reshape(sparse_bit_vector.size(0), 1)
        vector = sparse_bit_vector + 1
        # result = np.full(self.num_perm, 1, dtype=np.int)
        results = []
        for i in range(self.num_minhash):
            x = torch.matmul(vector, self.A1s[i]) + \
                        torch.matmul(torch.square(vector), self.A2s[i]) + \
                        torch.matmul(torch.pow(vector, 3), self.A3s[i]) + self.Bases[i]
            x = x % self.BIGPRIME % self.input_size
            x, _ = torch.min(x, axis=0)

            results.append(x)

        return (torch.matmul(self.A4s, torch.stack(results, 0))[0].data.numpy() + self.B4) % self.BIGPRIME % self.input_size



if __name__ == "__main__":
    gen = SparseBitVectorMinHashGenerator(1024,8)

    i = torch.LongTensor([[0, 0], [0, 1], [0, 2], [1, 1], [1, 3], [1, 4], [2, 3]])
    v = torch.ones(7)
    x = torch.sparse.FloatTensor(i.t(), v, torch.Size([3, 1024]))

    line0 = torch.index_select(x, 0, torch.LongTensor([0]))
    print(line0)
    print(line0._indices()[:, 1].data.numpy())
    print("x type: ", type(line0._indices()[:, 1]))
    # embedding0 = gen.generate(line0._indices()[:, 1]) #.data.numpy()
    embedding0 = gen.generate([0, 1])
    print("embedding0:", embedding0)

    line1 = torch.index_select(x, 0, torch.LongTensor([1]))
    print(line1)
    print(line1._indices()[:, 1].data.numpy())
    embedding1 = gen.generate(line1._indices()[:, 1]) # .data.numpy()
    print("embedding1:", embedding1)

    # intersection = [i for i in embedding0 if i in embedding1]
    intersection = 0
    for idx in range(len(embedding0)):
        if embedding0[idx] == embedding1[idx]:
            intersection += 1
    print(intersection/len(embedding0))
