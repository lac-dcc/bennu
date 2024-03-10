mkdir -p log
mkdir -p results

NAME="rtx3080"

BENCH=(
    matmul
    conv2d
    depthwise
    pooling
    reduce
    relu
)

for ((i = 0; i < ${#BENCH[@]}; i++)); do
    echo "Executing "${BENCH[i]}"..."
    python3 src/dpansor.py -m ansor -a cuda -t 1000 -l log/${BENCH[i]}.log -b ${BENCH[i]} 
done