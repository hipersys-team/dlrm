#!/bin/bash

chr() {
  [ "$1" -lt 256 ] || return 1
  printf "\\$(printf '%03o' "$1")"
}

ord() {
  LC_CTYPE=C printf '%d' "'$1"
}

export WORLD_SIZE=12
export RANK=$(( $(ord $(echo ${HOSTNAME:0:1})) - $(ord a) ))

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

eval "$($HOME/miniconda/bin/conda shell.bash hook)"
conda init
conda activate torch

if [[ $RANK != 0 ]]; then
  sleep 5
else
  printenv
fi

python3 dlrm_s_pytorch.py --arch-mlp-bot 2048-2048-2048-2048-2048-2048-2048-2048 --arch-mlp-top 1024-1024-1024-1024-1 --arch-interaction-op cat  --arch-embedding-size 10000000-10000000-10000000-10000000-10000000-10000000-10000000-10000000-10000000-10000000-10000000-10000000 --mini-batch-size 6144 --nepochs 5 --num-batches 16384 --use-gpu

