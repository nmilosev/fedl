#!/bin/bash

set -e

SERVER_ADDRESS="192.168.1.11:8080"
NUM_CLIENTS=5
NUM_EPOCHS=2
DOCKER_EXECUTABLE=podman
IMAGE_NAME=marvel-client

echo "Starting $NUM_CLIENTS clients."
for ((i = 0; i < $NUM_CLIENTS; i++))
do
    echo "Starting client(cid=$i) container with partition $i out of $NUM_CLIENTS clients."
    (
    $DOCKER_EXECUTABLE run \
      -e CID=$i \
      -e SERVER_ADDRESS=$SERVER_ADDRESS \
      -e NB_CLIENTS=$NUM_CLIENTS \
      -e EPOCHS=$NUM_EPOCHS \
      $IMAGE_NAME &
    )
      
done
echo "Started $NUM_CLIENTS client containers."
