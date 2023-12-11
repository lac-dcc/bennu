#!/bin/bash

MACHINE="RTX3080"
ARCH="cuda"

BENCH=(
    resnet-18
    resnet-34
    resnet-50
    resnet-101
    resnet-152
    mobilenet
    inception_v3
    squeezenet_v1.1
    #mxnet
    #bert
)

trials=10000

for ((i = 0; i < ${#BENCH[@]}; i++)); do
    echo "BENCH: "${BENCH[i]} 
    python3 benchmarks/models.py -m ansor -a $ARCH -t $trials -l results/$MACHINE"_"${BENCH[i]}_10k.json -b ${BENCH[i]}
done



