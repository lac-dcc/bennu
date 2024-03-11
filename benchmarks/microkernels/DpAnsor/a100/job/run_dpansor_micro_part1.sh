#!/bin/bash

#SBATCH --job-name=part1_dpansor_micro_cuda_a100
#SBATCH --output=./log/cuda_a100_micro_dp_part1.log
#SBATCH --mail-type=ALL --mail-user=gaurav.verma@stonybrook.edu
#SBATCH -p gpu --gpus=1  -C a100-80gb,ib -t 150:00:00
##SBATCH -p gpu --gpus=4  -C v100
#SBATCH -n 1 -c 32 --gpus-per-task=1

date
hostname
echo "Greetings from $SLURM_ARRAY_TASK_ID!"

# load modules
module load modules/2.1.1-20230405 
module load cmake/3.25.1  python/3.8.15 cuda/11.8.0  llvm/11.1.0 cudnn/8.4.0.27-11.6   gcc/10.4.0


export TVM_HOME=/mnt/home/gverma/ceph/tvm
export PYTHONPATH=/mnt/sw/nix/store/i613v246n7m0f6k22a8bwxsj51d1z6gb-llvm-11.1.0/lib/python3/site-packages:/mnt/home/gverma/ceph/tvm/python

TOP=(
    1
    10
    25
    50
    100
    200
    300
    1000
)

NAME="a100"

BENCH=(
    matmul
    relu
)

#for ((i = 0; i < ${#BENCH[@]}; i++)); do
#    echo "Executing "${BENCH[i]}"..."
#    python3 src/dpansor.py -m ansor -a cuda -t 1000 -l log/$NAME"_"${BENCH[i]}.log -b ${BENCH[i]}
#done

for ((j = 0; j < ${#TOP[@]}; j++)); do
    echo "Top-"${TOP[j]}
    RESULT=results/$NAME"_top"${TOP[j]}".csv"
    echo "Top-"${TOP[j]} >> $RESULT
    echo "bench, avg (ms), std (ms), trials, time total (min), ansor exec (ms), ansor tuning, speedup" >> $RESULT
    for ((i = 0; i < ${#BENCH[@]}; i++)); do
        echo "Executing "${BENCH[i]}"..."
        python3 src/dpansor.py -m dpansor -a cuda -l log/$NAME"_"${BENCH[i]}.log -k ${TOP[j]} -b ${BENCH[i]} >> $RESULT
    done
done

date
echo -e "\nCompleted\n"

