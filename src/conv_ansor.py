import sys, os
import numpy as np
import time
import tvm
from tvm import te, auto_scheduler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.kernels.conv import conv2d_ansor
from src.module import utils


if __name__ == "__main__":
    arch = "cpu"

    if len(sys.argv) == 2:
        arch = sys.argv[1]

    if arch == "cpu":
        target = tvm.target.Target("llvm")
        dev = tvm.cpu()
    elif arch == "cuda":
        target = tvm.target.Target("cuda")
        dev = tvm.cuda()
    elif arch == "arm":
        target = tvm.target.Target("llvm -mcpu=a64fx")
        dev = tvm.cpu()
    else:
        print("Archtecture doesn't support.")
        exit(0)

    print("Arch:", arch)

    input_shape = (1, 3, 224, 224)
    filter_shape = (64, 3, 3, 3)

    ## Create the search task
    task = tvm.auto_scheduler.SearchTask(
        func=conv2d_ansor, args=(input_shape, filter_shape), target=target
    )

    # Inspect the computational graph
    # print("Computational DAG:", task.compute_dag)

    ## Set Parameters for Auto-Scheduler
    log_file = f"results/{arch}_conv2d.json"

    if os.path.isfile(log_file):
        os.remove(log_file)

    trial = 100
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=trial,  # change this to 20000 to achieve the best performance
        runner=auto_scheduler.LocalRunner(number=10, repeat=3),
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=0,
    )

    ## Run the search

    start = time.time()
    # Run auto-tuning (search)
    task.tune(tune_option)
    end = time.time()

    time_avg, best_cfg = utils.get_best_time(log_file)

    print("Time spent:", time_avg)
    print("Config:", best_cfg)
    print("Time spent to search:", end - start)
    print("Time approximately for each search:", (end - start) / trial)
