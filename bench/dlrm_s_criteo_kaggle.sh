#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
#WARNING: must have compiled PyTorch and caffe2

#check if extra argument is passed to the test
if [[ $# == 1 ]]; then
    dlrm_extra_option=$1
else
    dlrm_extra_option=""
fi
#echo $dlrm_extra_option

run_pytorch=1
run_caffe2=0
nepochs=5 # 5

dlrm_pt_bin="python dlrm_s_pytorch.py"
dlrm_c2_bin="python dlrm_s_caffe2.py"

# --rand-hash-emb-flag --rand-hash-compression-rate=0.5
# --arch-embedding-size=1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000
# --arch-sparse-feature-size=64 --arch-mlp-bot=512-512-64 --arch-mlp-top=1024-1024-1024-1
# --data-generation=random --mini-batch-size=2048 --num-batches=1000 --num-indices-per-lookup=100
if [ $run_pytorch = 1 ]; then
    echo "run pytorch ..."
    # WARNING: the following parameters will be set based on the data set
    # --arch-embedding-size=... (sparse feature sizes)
    # --arch-mlp-bot=... (the input to the first layer of bottom mlp) 
#    --md-flag  --qr-flag --qr-threshold=20000   --rand-hash-emb-flag --rand-hash-compression-rate=0.03125 
    $dlrm_pt_bin --lsh-emb-flag --lsh-emb-compression-rate=0.03125 --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" --data-generation=dataset --data-set=kaggle --raw-data-file=./input/train.txt --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=2048 --nepochs=$nepochs --print-freq=1024 --print-time --use-gpu --test-mini-batch-size=16384 --test-num-workers=16 --test-freq=1000 --mlperf-logging --save-model=model.dat $dlrm_extra_option 2>&1 | tee run_kaggle_pt.log
fi

if [ $run_caffe2 = 1 ]; then
    echo "run caffe2 ..."
    # WARNING: the following parameters will be set based on the data set
    # --arch-embedding-size=... (sparse feature sizes)
    # --arch-mlp-bot=... (the input to the first layer of bottom mlp)
    $dlrm_c2_bin --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" --data-generation=dataset --data-set=kaggle --raw-data-file=./input/train.txt --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=128 --print-freq=1024 --print-time $dlrm_extra_option 2>&1 | tee run_kaggle_c2.log
fi

echo "done"
