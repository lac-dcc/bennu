#!/bin/bash

NAME="test" # change this
ARCH="cuda"
trials=1000

BENCH=(
    alexnet
    densenet121
    densenet161
    densenet169
    densenet201
    inception_v3
    mnasnet1_0
    mobilenet_v2
    resnet101
    resnet152
    resnet18
    googlenet
    resnet34
    resnet50
    shufflenet
    squeezenet
    vgg11
    vgg13
    vgg16
    vgg19
)

mkdir -p results
mkdir -p results/ms

for ((i = 0; i < ${#BENCH[@]}; i++)); do
    echo "BENCH: "${BENCH[i]} 
    DIR=results/ms/meta_$ARCH"_"$NAME"_"${BENCH[i]}_1k
    mkdir -p $DIR
    python3 benchmarks/models_onnx.py -m meta -a $ARCH -t $trials -l $DIR -b models/${BENCH[i]}.onnx &> $DIR/output.txt
done
