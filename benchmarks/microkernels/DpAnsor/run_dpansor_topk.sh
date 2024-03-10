mkdir -p log
mkdir -p results

NAME="rtx3080"

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

BENCH=(
    matmul
    conv2d
    depthwise
    pooling
    reduce
    relu
)

for ((j = 0; j < ${#TOP[@]}; j++)); do
    echo "Top-"${TOP[j]}
    RESULT=results/$NAME"_top"${TOP[j]}".csv"
    echo "Top-"${TOP[j]} > $RESULT
    echo "bench, avg (ms), std (ms), trials, time total (min), ansor exec (ms), ansor tuning, speedup" >> $RESULT
    for ((i = 0; i < ${#BENCH[@]}; i++)); do
        echo "Executing "${BENCH[i]}"..."
        python3 src/dpansor.py -m dpansor -a cuda -l log/$NAME"_"${BENCH[i]}.log -k ${TOP[j]} -b ${BENCH[i]} >> $RESULT
    done
done