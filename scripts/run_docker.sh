#!/bin/bash

help() 
{
    echo "Syntax: bash scripts/run_docker.sh [option]"
    echo "option:" 
    echo "  x86, arm, or cuda"
    exit 1
}

## MAIN
if [ $# -lt 1 ]; then
    help
else
    oper=$1
    if [ $oper == "help" ]; then
        help # call helper function
    elif [ $oper == "x86" ]; then
        docker run -v $PWD:/root/bennu -ti bennu-x86
    elif [ $oper == "arm" ]; then
        docker run -v $PWD:/root/bennu -ti bennu-arm32v7
    elif [ $oper == "cuda" ]; then
        docker run -v $PWD:/root/bennu -it --gpus all bennu-cuda
    fi
fi
