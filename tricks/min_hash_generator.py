import torch
import numpy as np
from typing import Set, Any

# minhash generator (based on permutation)
class SparseBitVectorMinHashGenerator:
    def __init__(self,
                 input_size,
                 num_perm=128):
        self.input_size = input_size
        self.num_perm = num_perm
        self.permutations = [np.random.permutation(self.input_size) for _ in range(self.num_perm)]

    def generate(self, sparse_bit_vector):
        idx_set: Set[Any] = set(sparse_bit_vector)
        result = np.zeros(self.num_perm, dtype=np.int)
        for result_idx, permutation in enumerate(self.permutations):
            for idx in range(self.input_size):
                if permutation[idx] in idx_set:
                    result[result_idx] = idx + 1
                    break
        return result


if __name__ == "__main__":
    gen = SparseBitVectorMinHashGenerator(1024,128)

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