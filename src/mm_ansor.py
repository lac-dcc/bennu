import sys, os
import numpy as np
import time
import tvm
from tvm import te, auto_scheduler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.kernels.mm import ansor_mm
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

    ## Create the search task

    N, L, M = 1000, 800, 700

    np.random.seed(0)
    a_np = np.random.uniform(size=(N, L)).astype(np.float32)
    b_np = np.random.uniform(size=(L, M)).astype(np.float32)
    c_np = a_np.dot(b_np)

    task = tvm.auto_scheduler.SearchTask(
        func=ansor_mm, args=(N, L, M, "float32"), target=target
    )

    # Inspect the computational graph
    # print("Computational DAG:", task.compute_dag)

    ## Set Parameters for Auto-Scheduler
    log_file = f"../results/{arch}_matmul.json"
    trial = 1000
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

    """
    # Apply the best schedule
    sch, args = task.apply_best(log_file)

    ## Check correctness and evaluate performance
    with auto_scheduler.ApplyHistoryBest(log_file):
        func = tvm.build(sch, args, target)
        a_tvm = tvm.nd.array(a_np, device=dev)
        b_tvm = tvm.nd.array(b_np, device=dev)
        c_tvm = tvm.nd.array(c_np, device=dev)
        func(a_tvm, b_tvm, c_tvm)

    # Check results
    #np.testing.assert_allclose(c_np, c_tvm.numpy(), rtol=1e-3)

    # Evaluate execution time.
    #evaluator = func.time_evaluator(func.entry_name, dev, number=10, repeat=3)
    #eval = evaluator(a_tvm, b_tvm, c_tvm)
    #print(", %f, %f, %f" % (eval.mean, eval.std, end-start))

    #print("Equivalent python schedule:")
    #print(task.print_best(log_file))
    """

    time_avg, best_cfg = utils.get_best_time(log_file)

    print("Time spent:", time_avg)
    print("Config:", best_cfg)
    print("Time spent to search:", end - start)
    print("Time approximately for each search:", (end - start) / trial)
