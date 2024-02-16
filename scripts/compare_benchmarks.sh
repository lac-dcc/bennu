#!/bin/bash

ARCH="x86"
TRIALS=100
NAME="Ryzen3700X"

#set -x

BENCH=(
    alexnet
    densenet121
    densenet161
    densenet169
    densenet201
    googlenet
    inception_v3
    mnasnet1_0
    mobilenet_v2
    resnet101
    resnet152
    resnet18
    resnet34
    resnet50
    shufflenet
    squeezenet
    vgg11
    vgg13
    vgg16
    vgg19
)

SIZE=(
    1
    10
    25
    50
    100
    200
    300
    1000
)

mkdir -p "perf_stats/"$NAME
for ((i = 0; i < ${#BENCH[@]}; i++)); do
    RES="perf_stats/$NAME/"${BENCH[i]}".csv"
    echo "BENCH: "${BENCH[i]} > $RES
    for ((j = 0; j < ${#SIZE[@]}; j++)); do
        echo "SIZE: "${SIZE[j]}
        python3 benchmarks/models_onnx_compare.py -m grid -a $ARCH -t $TRIALS -k ${SIZE[j]} -l results/$ARCH"_"${BENCH[i]}_10k.json -b models/${BENCH[i]}.onnx >> $RES
    done
done