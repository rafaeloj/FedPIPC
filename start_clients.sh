#!/bin/bash

# utilizando a biblioteca python yq para acessar os dados do environment.yaml
N_CLIENTS=$(($(yq '.N_CLIENTS' environment.yaml)-1))
for i in $(seq 0 $N_CLIENTS)
do
    python3 client.py --cid=$i &
done