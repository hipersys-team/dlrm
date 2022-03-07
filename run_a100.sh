#!/bin/bash

export WORLD_SIZE=12
export RANK=0
export MASTER_PORT=27182
export MASTER_ADDR="abtin.csail.mit.edu"

#export NCCL_IB_DISABLE=1
#export NCCL_SOCKET_IFNAME="enp3s0f0"
#export NCCL_DEBUG="INFO"

export NCCL_NET="IB"
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA="qedr0,qedr1,qedr2,qedr3"
export LD_LIBRARY_PATH="/home/frankwwy/mccl/build/lib:"
export LD_PRELOAD="/home/frankwwy/mccl/build/lib/libnccl.so:"
export NCCL_MIN_NCHANNELS=2
export NCCL_MAX_NCHANNELS=2
export NCCL_ALGO=Ring

python3 dlrm_s_pytorch.py  --arch-embedding-size 10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000 --mini-batch-size 12 --nepochs 10 --num-batches 16 --use-gpu

