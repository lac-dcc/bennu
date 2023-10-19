import argparse
import json
from pathlib import Path

import tvm
from tvm import auto_scheduler
from tvm.auto_scheduler.measure import recover_measure_input

#ToDo: ansor_mm will be removed once the task is saved as pkl
import tvm.te as te

@auto_scheduler.register_workload
def ansor_mm(N, L, M, dtype="float32"):
    A = te.placeholder((N, L), name="A", dtype=dtype)
    B = te.placeholder((L, M), name="B", dtype=dtype)

    k = te.reduce_axis((0, L), name="k")
    C = te.compute((N, M), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")

    return [A, B, C]

def print_measurement_record_info(index, input, result):
    input = recover_measure_input(input, True)
    print(f"Index: {index}")
    print(f"Time cost (second): {result.costs}")
    print(f"-"*10)
    print(f"Workload Key: {input.task.workload_key}")
    print(f"FLOP Ct: {input.task.compute_dag.flop_ct}")
    print(f"-"*10)
    print(f"Compute DAG:")
    print(input.task.compute_dag)
    print(f"Program:")
    print(input.state)
    print(f"\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("python print_record_info.py --logfile '../results/cpu_matmul.json' --index 3")
    parser.add_argument("--logfile", type=str, required=True)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()

    file_path = args.logfile
    path = Path(file_path)

    inputs, results = [], []

    try:
        if path.is_file():
            print(f"The file {file_path} exists.")
        with open(file_path, "r") as file:
            inputs, results = auto_scheduler.RecordReader(file_path).read_lines()
    except FileNotFoundError as e:
        print(e)

    idx = args.index

    #ToDo: while running mm_ansor, tasks should be stored in pkl format to be used here
    # to register the tasks
    # workload_key = ["ansor_mm", 1000, 1000, 1000, "float32"]
    # compute_dag_tensors = [Tensor(shape=[1000, 1000], op.name=A), Tensor(shape=[1000, 1000], op.name=B), Tensor(shape=[1000, 1000], op.name=C)]
    task = tvm.auto_scheduler.SearchTask(
        func=ansor_mm, args=(1000, 1000, 1000, "float32"), target=tvm.target.Target("llvm")
    )
    
    auto_scheduler.workload_registry.register_workload_tensors(task.workload_key, task.compute_dag.tensors)
    
    print_measurement_record_info(idx, inputs[idx], results[idx])
