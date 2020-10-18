# data preprocessing for LSH embedding
import numpy as np
import torch
from tricks.min_hash_generator import SparseBitVectorMinHashGenerator
from collections import defaultdict
# import multiprocessing
from tqdm import tqdm
import time
# from p_tqdm import p_map

# cores = multiprocessing.cpu_count()
# pool = multiprocessing.Pool(processes=cores)
# print(cores)

def getMinHashTable():
    data = np.load('./input/kaggleAdDisplayChallenge_processed.npz')
    min_hash_tables = []
    start = time.time()
    print(len(data["X_cat"][0]))
    for fea_id in range(len(data["X_cat"][0])):
        print(fea_id)
        i = [[], []] # use torch tensor
        val_indices = defaultdict(lambda:[])
        cat_fea = data['X_cat'][:, fea_id]
        for doc_id in range(len(cat_fea)):
            val_indices[cat_fea[doc_id]].append(doc_id)
        print(len(val_indices), len(cat_fea))
        min_hash_gen = SparseBitVectorMinHashGenerator(len(cat_fea), 16)
        min_hash_table = []
        for val_id in range(len(val_indices)):
            min_hash_table.append(min_hash_gen.generate(val_indices[val_id]))
        min_hash_tables.append(min_hash_table)
        # print(min_hash_table[0])

    np.savez(r'./input/minHashTables.npz', *min_hash_tables)

    end = time.time()
    print(end - start)

if __name__ == "__main__":
    getMinHashTable()