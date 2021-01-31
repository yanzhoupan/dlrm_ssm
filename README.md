# Similarity-Based Shared Memory Embeddings for Recommendation Systems


## Description:

This project applys similarity-based shared memory(SSM) embeddings on DLRM. It is modified from Facebook's DLRM repository(https://github.com/facebookresearch/dlrm). 


## Implementation

**DLRM PyTorch**. We only support SSM for PyTorch currently.

       dlrm_s_pytorch.py


## How to run our code?
### prerequests:
pytorch-nightly (*6/10/19*)

scikit-learn

numpy

onnx (*optional*)

pydot (*optional*)

torchviz (*optional*)

cuda-dev(*optional*)

tqdm

### run SSM embeddings with DLRM:
1. Pre-training: use lsh_pp_pretraining.py to generate the min_hash_table for SSM. 3 hyperparameter is needed：
* EMBEDDING: embedding dimension
* NUM_HASH: number of hash functions used to generate min_hash_table
* NUM_PT: number of datapoints used to represent the dataset

A sample command is: 


#### command line arguments
|command line arguments|type|usage|
|:--------------------|:---|:----|
|mini-batch-size|int|batch size|
|nepochs|int|num of epochs|
|arch-sparse-feature-size|string|we use "13-512-256-128" which is the default value for DLRM|
|arch-mlp-bot|string|we use "1024-1024-512-256-1" which is the default value for DLRM|
|data-generation|string|please use "dataset" to run our code|
|data-set|string|please use "kaggle" to run our code|
|raw-data-file|string|please use "./input/train.txt" to run our code|
|processed-data-file|string|please use "./input/kaggleAdDisplayChallenge_processed.npz" to run our code|
|loss-function|string|please use "bce"|
|round-targets|boolean|pleause use "True"|
|learning-rate|double|our default learning rate is 0.1|
|print-freq|int|the frequency to print log|
|print-time|-||
|test-mini-batch-size|int|the batch size for test, please use 16384 to run our code| 
|test-num-workers|int|we use 16 for our experiment|


#### A sample run of the code
A sample that runs our code is in:
```
bench/dlrm_s_criteo_kaggle.sh
```



## License

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
