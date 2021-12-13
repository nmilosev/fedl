#!/bin/bash

set -e

SERVER_ADDRESS="[::]:8080"
NUM_CLIENTS=5
NUM_EPOCHS=2

echo "Starting $NUM_CLIENTS clients."
for ((i = 0; i < $NUM_CLIENTS; i++))
do
    echo "Starting client(cid=$i) with partition $i out of $NUM_CLIENTS clients."
    (
    python client.py \
      --cid=$i \
      --server_address=$SERVER_ADDRESS \
      --nb_clients=$NUM_CLIENTS \
      --epochs=$NUM_EPOCHS &
    )
      
done
echo "Started $NUM_CLIENTS clients."
