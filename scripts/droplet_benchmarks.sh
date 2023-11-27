#!/bin/bash

ARCH="x86"
TRIALS=100

BENCH=(
    #alexnet
    #densenet121
    #densenet161
    #densenet169
    #densenet201
    googlenet
    #inception_v3
    #mnasnet1_0
    #mobilenet_v2
    #resnet101
    #resnet152
    #resnet18
    #resnet34
    #resnet50
    #shufflenet
    #squeezenet
    #vgg11
    #vgg13
    #vgg16
    #vgg19
)

SIZE=(
    #1
    #10
    #25
    #50
    #100
    #200
    300
    10000
)

for ((i = 0; i < ${#BENCH[@]}; i++)); do
    echo "BENCH: "${BENCH[i]} 
    for ((j = 0; j < ${#SIZE[@]}; j++)); do
        echo "SIZE: "${SIZE[j]}
        python3 benchmarks/models_onnx.py -m droplet -a $ARCH -t $TRIALS -k ${SIZE[j]} -l results/$ARCH"_"${BENCH[i]}_10k.json -b models/${BENCH[i]}.onnx
    done
done


