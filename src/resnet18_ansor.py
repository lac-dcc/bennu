import sys, os
import numpy as np
import time
import tvm
from tvm import relay, auto_scheduler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.kernels.resnet18 import resnet18_ansor
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

    # Inspect the computational graph
    # print("Computational DAG:", task.compute_dag)

    ## Set Parameters for Auto-Scheduler
    log_file = f"../results/{arch}_resnet18.json"

    if os.path.isfile(log_file):
        os.remove(log_file)
    
    batch_size = 1
    dtype = 'float32'
    tasks, task_weights, mod, params, data_shape, out_shape = resnet18_ansor(batch_size, target)

    trials = 30
    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=trials,  # change this to 20000 to achieve the best performance
        runner=auto_scheduler.LocalRunner(number=2, repeat=3, min_repeat_ms=100, enable_cpu_cache_flush=True if target=="llvm" else False),
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=0
    )

    ## Run the search
    start = time.time()
    tuner.tune(tune_option)
    end = time.time()

    # compile kernels in kernel tuned only mode
    with auto_scheduler.ApplyHistoryBest(log_file):
        with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
            lib = relay.build(mod, target=target, params=params)
            r = utils.evaluate_performance(lib, data_shape, target)

    print("Time for each layers")
    results = utils.get_best_time_multilayer(log_file)
    for i, v in enumerate(results):
        print(f"Layer {i}: Time {np.mean(results[v][0])} cfg: {results[v][1]}")
    
    print("\nTime to execute the algorithm: ", np.mean(r))
    print("Time spent to search:", end - start)
    print("Time approximately for each search:", (end - start) / trials)
