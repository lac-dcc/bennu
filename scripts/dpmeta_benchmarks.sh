#!/bin/bash

NAME="test" # change this
ARCH="x86"
trials=100

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

TOP=(
    1
    10
    50
    100
    200
    300
    1000
)

mkdir -p results
mkdir -p results/ms

for ((i = 0; i < ${#BENCH[@]}; i++)); do
    echo "BENCH: "${BENCH[i]} 
    DIR=results/ms/meta_$ARCH"_"$NAME"_"${BENCH[i]}_10k
    for ((j = 0; j < ${#TOP[@]}; j++)); do
        echo "TOP: "${TOP[j]}
        python3 benchmarks/models_onnx.py -m dpmeta -a $ARCH -t $trials -k ${TOP[j]} -l $DIR -b models/${BENCH[i]}.onnx 
    done
done
