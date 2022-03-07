#!/bin/bash

export WORLD_SIZE=1
export RANK=0
export MASTER_PORT=27182
export MASTER_ADDR="abtin.csail.mit.edu"

export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME="enp3s0f0"
export NCCL_DEBUG="INFO"

python3 dlrm_s_pytorch.py  --arch-embedding-size 10000-10000 --nepochs 10 --num-batches 16 --use-gpu

