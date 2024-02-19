#!/bin/bash

ARCH="x86"
TRIALS=100
NAME="Ryzen3700X"

set -x

BENCH=(
    alexnet
    vgg11
    resnet18
    resnet34
    vgg13
    vgg16
    vgg19
    shufflenet
    squeezenet
    resnet101
    resnet152
    resnet50
    mobilenet_v2
    mnasnet1_0
    inception_v3
    googlenet
    densenet121
    densenet161
    densenet169
    densenet201
)

SIZE=(
    300
)

METHOD=(
    #grid
    random
    #ga
    #xgb
)

mkdir -p "perf_stats/"$NAME

for ((k = 0; k < ${#METHOD[@]}; k++)); do
    echo ${METHOD[k]}
    mkdir -p "perf_stats/$NAME/"${METHOD[k]}
    for ((i = 0; i < ${#BENCH[@]}; i++)); do
        RES="perf_stats/$NAME/"${METHOD[k]}/${BENCH[i]}".csv"
        echo "BENCH: "${BENCH[i]} > $RES
        for ((j = 0; j < ${#SIZE[@]}; j++)); do
            echo "SIZE: "${SIZE[j]}
            python3 benchmarks/models_onnx_compare.py -m ${METHOD[k]} -a $ARCH -t $TRIALS -k ${SIZE[j]} -l results/$ARCH"_"${BENCH[i]}_10k.json -b models/${BENCH[i]}.onnx >> $RES
        done
    done
done