# data preprocessing for LSH embedding
import numpy as np
import torch
from min_hash_generator import SparseBitVectorMinHashGenerator
from collections import defaultdict
# import multiprocessing
from tqdm import tqdm
import time
import random
import concurrent.futures
import pdb

seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


# use partial data set to get minhash table.
min_hash_gen = None
val_indices = None
#EMBEDDING = 256
#NUM_HASH = 2

import sys

if len(sys.argv) <=1:
    print("Usage: <script> embedding hash num_pt")
    assert(False)
EMBEDDING = int(sys.argv[1])
NUM_HASH = int(sys.argv[2])
NUM_PT = int(sys.argv[3])
print("EMB:",EMBEDDING, "NUMH",NUM_HASH, "NUM_PT",NUM_PT)

def compute(start, end):
    global min_hash_table
    p_min_hash_table = np.zeros((end-start, EMBEDDING))
    for val_id in range(start, end):
        p_min_hash_table[val_id-start] = min_hash_gen.generate(val_indices[val_id])

    return start,end ,p_min_hash_table

def getBigMinHashTable():
    global min_hash_gen, min_hash_table, val_indices
    data = np.load('./input/kaggleAdDisplayChallenge_processed.npz')
    data_num, cat_num = data["X_cat"].shape # (45840617, 26) for criteo
    partial_idx = np.random.choice(np.arange(data_num), size=NUM_PT, replace=False)
    partial_cat_data = data['X_cat'][partial_idx]
    print(partial_cat_data.shape)


    start_time = time.time()
    np.savez(r'./cat_counts.npz', cat_counts = data['counts'])

    base = 0
    val_indices = defaultdict(lambda:[])
    # generate signiture matrix for category values (partial data)
    for fea_id in tqdm(range(cat_num)):
        cat_fea = partial_cat_data[:, fea_id]
        
        for doc_id in range(len(cat_fea)): # loop over docs
            val_indices[cat_fea[doc_id] + base].append(doc_id)

        for val in range(data['counts'][fea_id]):
            if val_indices[val+base] == []: 
                val_indices[val+base] = [random.randint(0, data_num+1)] # set val_indices to a random place if never seen it
        base += data['counts'][fea_id]
    
    embedding_dim = EMBEDDING
    min_hash_table = np.zeros((len(val_indices), embedding_dim))
    input_size = len(cat_fea) # number of the data items
    min_hash_gen = SparseBitVectorMinHashGenerator(input_size, embedding_dim, NUM_HASH)

    batch_size=1000
    with concurrent.futures.ProcessPoolExecutor(50) as executor:
        print("submitting jobs")
        futures = []
        print ("total", len(val_indices))
        total = len(val_indices)
        num_batches = int(np.ceil(len(val_indices) / batch_size))
        for i in tqdm(range(num_batches)):
          start = i * batch_size
          end = min(total, start + batch_size)
          if end > start:
              futures.append(executor.submit(compute, start, end))
          #compute(start, end)
        ip = 0
        for res in tqdm(concurrent.futures.as_completed(futures), total = num_batches):
          st,ed,output = res.result()
          ip = ip + 1
          min_hash_table[st:ed,:] = output
          #print(st, ed, np.sum(min_hash_table[st:ed]))
    np.savez(r'./input/bigMinHashTable_H'+ str(NUM_HASH) + '_E' + str(EMBEDDING)+ '_P' + str(NUM_PT) + '.npz', big_min_hash_table = min_hash_table.astype(int))

    end_time = time.time()
    print(end_time - start_time)


if __name__ == "__main__":
    # getMinHashTable()
    getBigMinHashTable()
    # bigMinHashTable = np.load('./input/bigMinHashTable.npz')
    # minHashTables = np.load('./input/minHashTables.npz')
    # print(len(minHashTables['arr_0'][:, 0]))
    # print(len(bigMinHashTable['big_min_hash_table'][:, 0]))
