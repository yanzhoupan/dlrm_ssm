# data preprocessing for LSH embedding
import numpy as np
import torch
from min_hash_generator import SparseBitVectorMinHashGenerator
from collections import defaultdict
import multiprocessing
from tqdm import tqdm
from p_tqdm import p_map

def getMinHashTable(fea_id):
    # print(fea_id)
    data = np.load('./dlrm/input/kaggleAdDisplayChallenge_processed.npz')
    i = [[], []] # use torch tensor
    val_indices = defaultdict(lambda:[])
    cat_fea = data['X_cat'][:, fea_id]
    for doc_id in range(len(cat_fea)):
        val_indices[cat_fea[doc_id]].append(doc_id)
    # print(len(val_indices))
    
    min_hash_gen = SparseBitVectorMinHashGenerator(len(cat_fea), 16)
    min_hash_table = []
    for val_id in range(len(val_indices)):
        min_hash_table.append(min_hash_gen.generate(val_indices[val_id]))
    # print(min_hash_table[0])
    return min_hash_table
    

if __name__ == '__main__':
    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cores)
    fea_ids = range(26)
    res = list(tqdm(pool.imap(getMinHashTable, fea_ids),total=len(fea_ids),desc='Minhash table generating: '))
    np.savez(r'./dlrm/input/minHashTables.npz', *res)
