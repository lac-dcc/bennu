#!/bin/bash

NAME="GTX1650"
ARCH="cuda"

BENCH=(
    #alexnet
    #densenet121
    #densenet161
    #densenet169
    #densenet201
    #inception_v3
    #mnasnet1_0
    #mobilenet_v2
    #resnet101
    #resnet152
    resnet18
    #googlenet
    #resnet34
    #resnet50
    #shufflenet
    #squeezenet
    #vgg11
    #vgg13
    #vgg16
    #vgg19
)

trials=1000

for ((i = 0; i < ${#BENCH[@]}; i++)); do
    echo "BENCH: "${BENCH[i]} 
    python3 benchmarks/models_onnx.py -m ansor -a $ARCH -t $trials -l results/$NAME"_"${BENCH[i]}_1k.json -b models/${BENCH[i]}.onnx
done
