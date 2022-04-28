#!/bin/bash

set -e

SERVER_ADDRESS="[::]:8080"
NUM_CLIENTS=4
NUM_EPOCHS=5

echo "Starting $NUM_CLIENTS clients."
for ((i = 0; i < $NUM_CLIENTS; i++))
do
    echo "Starting client(cid=$i) with partition $i out of $NUM_CLIENTS clients."
    (
    python client.py \
      --cid=$i \
      --server_address=$SERVER_ADDRESS \
      --nb_clients=$NUM_CLIENTS \
      --epochs=$NUM_EPOCHS &> client_$i.log &
    )
      
done
echo "Started $NUM_CLIENTS clients."
