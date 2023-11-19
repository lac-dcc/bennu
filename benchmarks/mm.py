import os, sys, time, argparse
from tvm import te, autotvm, auto_scheduler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.DropletSearch import Droplet
from src.utils import *

## ------------------ Global ---------------------
N, L, M = 1000, 800, 700
dtype = "float32"


## ----------------- Benchmark -------------------
@auto_scheduler.register_workload
def ansor_mm(N, L, M, dtype="float32"):
    A = te.placeholder((N, L), name="A", dtype=dtype)
    B = te.placeholder((L, M), name="B", dtype=dtype)

    k = te.reduce_axis((0, L), name="k")
    C = te.compute((N, M), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")

    return [A, B, C]

## ---------------------------------------------


def generate_ansor_template(log_file, target, trials):
    task = tvm.auto_scheduler.SearchTask(
        func=ansor_mm, args=(N, L, M, "float32"), target=target
    )

    ## Set Parameters for Auto-Scheduler
    trial = trials
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=trial,  # change this to 20000 to achieve the best performance
        runner=auto_scheduler.LocalRunner(
            number=10,
            repeat=3,
            timeout=100,
            enable_cpu_cache_flush=True if target == "llvm" else False,
        ),
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=0,
    )

    start = time.time()
    # Run auto-tuning (search)
    task.tune(tune_option)
    end = time.time()

    time_avg, best_cfg = get_best_time(log_file)

    print("Time spent:", time_avg)
    print("Config:", best_cfg)
    print("Time spent to search:", end - start)


def build_template(log_file, target, trials):
    t_ansor, workload, json_file = get_best_template(log_file)

    print("Layer, Time Droplet (s), Tuning time Droplet (s), tasks Droplet, Time Ansor (s), tasks Ansor, speedup")

    log = "mm.log"
    clean_file(log)

    droplet = Droplet(json_file, workload, target, log, trials)
    start = time.time()
    droplet.tune()
    end = time.time()

    droplet_avg, droplet_cfg = get_best_time(log)
            
    print(
        "%.7f, %.2f, %d, %.7f, %d, %.2f"
        % (
            np.mean(droplet_avg),
            end - start,
            get_tasks(log),
            np.mean(t_ansor),
            get_task_multilayers(logfile)[workload],
            np.mean(t_ansor) / np.mean(droplet_avg),
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "python mm.py -m 'ansor' -a x86 -l 'results/cpu_matmul.json' -i 3"
    )
    parser.add_argument(
        "-m", "--method", type=str, required=True, help="Options: ansor, droplet"
    )
    parser.add_argument(
        "-a", "--arch", type=str, required=True, help="Options: x86, aarch64, cuda"
    )
    parser.add_argument("-l", "--logfile", type=str, required=True)
    parser.add_argument("-t", "--trials", type=int, default=100)
    args = parser.parse_args()

    method = args.method
    arch = args.arch
    logfile = args.logfile
    trials = args.trials

    if arch == "x86":
        target = tvm.target.Target("llvm")
        dev = tvm.cpu()
    elif arch == "cuda":
        target = tvm.target.Target("cuda")
        dev = tvm.cuda()
    elif arch == "aarch64":
        target = tvm.target.Target("llvm -mcpu=a64fx")
        dev = tvm.cpu()
    else:
        print("Archtecture doesn't support.")
        exit(0)

    if method == "ansor":
        generate_ansor_template(logfile, target, trials)
    elif method == "droplet":
        build_template(logfile, target, trials)
