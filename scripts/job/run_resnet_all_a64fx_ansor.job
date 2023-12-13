#!/usr/bin/bash

#SBATCH --job-name=run_resnet_all_a64fx_ansor
#SBATCH -p long -t 48:00:00
#SBATCH --output=./logs/a64fx/run_resnet101_a64fx_ansor.log
#SBATCH -N 1 -n 48
###SBATCH --mail-type=ALL
#SBATCH --mail-user=gaurav.verma@stonybrook.edu

date
hostname

module purge
module load slurm/slurm/19.05.7  # anaconda/3  llvm/16.0.5

source /lustre/software/anaconda3/aarch64/etc/profile.d/conda.sh
conda activate tvm
cd /lustre/projects/ML-group/gverma/bennu

export TVM_HOME=/lustre/projects/ML-group/gverma/tvm
export PYTHONPATH=/lustre/projects/ML-group/gverma/tvm/python

echo "python" `which python`
echo "TVM_HOME" $TVM_HOME
echo "PYTHONPATH" $PYTHONPATH
echo `module li`

# alexnet.onnx      densenet201.onnx   mobilenet_v2.onnx  resnet34.onnx    vgg11.onnx
# densenet121.onnx  googlenet.onnx     resnet101.onnx     resnet50.onnx    vgg13.onnx
# densenet161.onnx  inception_v3.onnx  resnet152.onnx     shufflenet.onnx  vgg16.onnx
# densenet169.onnx  mnasnet1_0.onnx    resnet18.onnx      squeezenet.onnx  vgg19.onnx

/lustre/home/gverma/.conda/envs/tvm/bin/python benchmarks/models_onnx.py -m ansor -a arm -t 10000 -l results/a64fx_resnet101_10k.json -b models/resnet101.onnx

wait

# /lustre/home/gverma/.conda/envs/tvm/bin/python benchmarks/models_onnx.py -m ansor -a arm -t 10000 -l results/a64fx_resnet152_10k.json -b models/resnet152.onnx

# wait 

# /lustre/home/gverma/.conda/envs/tvm/bin/python benchmarks/models_onnx.py -m ansor -a arm -t 10000 -l results/a64fx_resnet34_10k.json -b models/resnet34.onnx

# wait

# /lustre/home/gverma/.conda/envs/tvm/bin/python benchmarks/models_onnx.py -m ansor -a arm -t 10000 -l results/a64fx_resnet50_10k.json -b models/resnet50.onnx

echo -e "\nCompleted\n"
