#!/bin/bash

chr() {
  [ "$1" -lt 256 ] || return 1
  printf "\\$(printf '%03o' "$1")"
}

ord() {
  LC_CTYPE=C printf '%d' "'$1"
}

export RANK=$(( $(ord $(echo ${HOSTNAME:0:1})) - $(ord a) + 1 ))

sudo ip link set dev enp194s0 up
sudo ip link set dev enp194s0 mtu 9000
sudo ip r a 192.168.0.0/24 via 192.168.$RANK.2

