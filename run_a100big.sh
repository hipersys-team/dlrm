#!/bin/bash

chr() {
  [ "$1" -lt 256 ] || return 1
  printf "\\$(printf '%03o' "$1")"
}

ord() {
  LC_CTYPE=C printf '%d' "'$1"
}

export WORLD_SIZE=2
export RANK=0 # 0 for master, 1 for worker
# $(( $(ord $(echo ${HOSTNAME:0:1})) - $(ord a) ))

export MASTER_PORT=27182
export MASTER_ADDR="abtin.csail.mit.edu"

#export NCCL_IB_DISABLE=1
#export NCCL_SOCKET_IFNAME="enp3s0f0"
#export NCCL_DEBUG="INFO"

export NCCL_NET="IB"
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA="mlx5_0"
#export NCCL_IB_HCA="qedr0,qedr1,qedr2,qedr3"
export LD_LIBRARY_PATH="/home/frankwwy/nccl/build/lib:"
export LD_PRELOAD="/home/frankwwy/nccl/build/lib/libnccl.so:"

runname=big
embedding_dim=4096
mlp_bot=8192-8192-8192-8192-8192-8192-8192-8192-8192-8192-8192-8192-8192-8192-8192-$embedding_dim
mlp_top=4096-4096-4096-4096-4096-4096-4096-4096-1
embedding_sz=1000000
prof=""
declare -a l_batch_szs=(256)

eval "$($HOME/miniconda/bin/conda shell.bash hook)"
conda init
conda activate torch

for l_batch_sz in "${l_batch_szs[@]}"; do
  if [[ $RANK != 0 ]]; then
    sleep 5
    python3 dlrm_s_pytorch.py --dist-backend="nccl" --arch-mlp-bot $mlp_bot --arch-sparse-feature-size $embedding_dim --arch-mlp-top $mlp_top --arch-interaction-op dot --arch-embedding-size ${embedding_sz}-${embedding_sz} --mini-batch-size $(( 2 * $l_batch_sz )) --nepochs 5 --num-batches 64 --use-gpu --print-time --dataset-multiprocessing $prof
  else
    printenv
    python3 dlrm_s_pytorch.py --dist-backend="nccl" --arch-mlp-bot $mlp_bot --arch-sparse-feature-size $embedding_dim --arch-mlp-top $mlp_top --arch-interaction-op dot --arch-embedding-size ${embedding_sz}-${embedding_sz} --mini-batch-size $(( 2 * $l_batch_sz )) --nepochs 5 --num-batches 64 --use-gpu --print-time --dataset-multiprocessing $prof > result_${runname}_${l_batch_sz}.res
  fi
  sleep 20
  killall ssh
done


