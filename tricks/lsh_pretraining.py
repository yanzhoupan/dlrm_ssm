# data preprocessing for LSH embedding
import numpy as np
import torch
from tricks.min_hash_generator import SparseBitVectorMinHashGenerator
from collections import defaultdict
# import multiprocessing
from tqdm import tqdm
import time

# def getMinHashTable():
#     data = np.load('./input/kaggleAdDisplayChallenge_processed.npz')
#     min_hash_tables = []
#     start = time.time()
#     cat_num = len(data["X_cat"][0]) # 26
#     # print()
#     for fea_id in tqdm(range(cat_num)):
  
#         val_indices = defaultdict(lambda:[])
#         cat_fea = data['X_cat'][:, fea_id]
#         for doc_id in range(len(cat_fea)):
#             val_indices[cat_fea[doc_id]].append(doc_id)

#         min_hash_gen = SparseBitVectorMinHashGenerator(len(cat_fea), 16) 
#         min_hash_table = []
#         for val_id in range(len(val_indices)):
#             min_hash_table.append(min_hash_gen.generate(val_indices[val_id])) # TODO: use val_sp as the input to generate
#         min_hash_tables.append(min_hash_table)

#     np.savez(r'./input/minHashTables.npz', *min_hash_tables)

#     end = time.time()
#     print(end - start)


def getBigMinHashTable_allData():
    data = np.load('./input/kaggleAdDisplayChallenge_processed.npz')
    start = time.time()
    cat_num = data["X_cat"].shape[1] # 26
    # print(data['X_cat'].shape, data.files)
    np.savez(r'./input/cat_counts.npz', cat_counts = data['counts'])

    base = 0
    val_indices = defaultdict(lambda:[])
    # generate signiture matrix for category values
    for fea_id in tqdm(range(cat_num)):
        cat_fea = data['X_cat'][:, fea_id]
        
        for doc_id in range(len(cat_fea)): # loop over docs
            val_indices[cat_fea[doc_id] + base].append(doc_id)
        base += data['counts'][fea_id]
    
    min_hash_table = []
    embedding_dim = 16
    input_size = len(cat_fea) # number of the data items
    min_hash_gen = SparseBitVectorMinHashGenerator(input_size, embedding_dim, 2)
    for val_id in range(len(val_indices)):
        min_hash_table.append(min_hash_gen.generate(val_indices[val_id]))

    np.savez(r'./input/bigMinHashTable.npz', big_min_hash_table = min_hash_table)

    end = time.time()
    print(end - start)


# use partial data set to get minhash table.
def getBigMinHashTable():
    data = np.load('./input/kaggleAdDisplayChallenge_processed.npz')
    data_num, cat_num = data["X_cat"].shape # (45840617, 26)
    ratio = 0.0028 # using 125k samples
    partial_idx = np.random.choice(np.arange(data_num), size=int(data_num * ratio), replace=False)
    partial_cat_data = data['X_cat'][partial_idx]


    start = time.time()
    np.savez(r'./input/cat_counts.npz', cat_counts = data['counts'])

    base = 0
    val_indices = defaultdict(lambda:[])
    # generate signiture matrix for category values
    for fea_id in tqdm(range(cat_num)):
        cat_fea = partial_cat_data[:, fea_id]
        
        for doc_id in range(len(cat_fea)): # loop over docs
            val_indices[cat_fea[doc_id] + base].append(doc_id)
        for val in range(data['counts'][fea_id]):
            if val_indices[val+base] == []: 
                val_indices[val+base] = [45840618] # set val_indices to a fixed place if never seen it
        base += data['counts'][fea_id]
    
    min_hash_table = []
    embedding_dim = 16
    input_size = len(cat_fea) # number of the data items
    min_hash_gen = SparseBitVectorMinHashGenerator(input_size, embedding_dim, 2)
    for val_id in range(len(val_indices)):
        min_hash_table.append(min_hash_gen.generate(val_indices[val_id]))

    np.savez(r'./input/bigMinHashTable.npz', big_min_hash_table = min_hash_table)

    end = time.time()
    print(end - start)


if __name__ == "__main__":
    # getMinHashTable()
    # getBigMinHashTable()
    bigMinHashTable = np.load('./input/bigMinHashTable.npz')
    minHashTables = np.load('./input/minHashTables.npz')
    print(len(minHashTables['arr_0'][:, 0]))
    print(len(bigMinHashTable['big_min_hash_table'][:, 0]))