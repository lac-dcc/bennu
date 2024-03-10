rm -rf log
mkdir -p log
mkdir -p results

TOP=(
    1
    #10
    #25
    #50
    #100
    #200
    #300
    #1000
)

BENCH=(
    #matmul
    #conv2d
    #depthwise
    #pooling
    reduce
    #relu
)

for ((i = 0; i < ${#BENCH[@]}; i++)); do
    echo "Executing "${BENCH[i]}"..."
    for ((j = 0; j < ${#TOP[@]}; j++)); do
        echo "Top-"${TOP[j]}
        python3 src/dpansor.py -a cuda -l log/${BENCH[i]}.log -t ${TOP[j]} -b ${BENCH[i]} > results/${BENCH[i]}.csv
        python3 src/print_output.py results/${BENCH[i]}.csv
    done
done