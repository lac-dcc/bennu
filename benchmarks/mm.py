import os, sys, time, argparse
from tvm import te, autotvm, auto_scheduler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.template_factory import Template_factory
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


@autotvm.template("autotvm_mm")
def autotvm_mm(N, K, M, dtype="float32", cfg=None):
    k = te.reduce_axis((0, K), "k")
    A = te.placeholder((M, K), name="A")
    B = te.placeholder((K, N), name="B")
    C = te.compute((M, N), lambda m, n: te.sum(A[m, k] * B[k, n], axis=k), name="C")

    if cfg is not None:
        return Template_factory(cfg, [A, B, C])
    else:
        return te.create_schedule(C.op), [A, B, C]


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


def build_template(log_file, index, target, trials):
    config = get_template_ansor(log_file)

    print(
        "N, Droplet Time (s), Search Time (s), Count Sample, Ansor Time (s), speedup (Ansor/Droplet)"
    )
    for i in range(len(config)):
        if i == index or index == -1:
            t_ansor, cfg_ansor = config[i]

            task = autotvm.task.create(
                "autotvm_mm", args=(N, L, M, "float32", cfg_ansor), target=target
            )

            # logging.getLogger("autotvm").setLevel(logging.DEBUG)
            # logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))

            measure_option = autotvm.measure_option(
                builder="local",
                runner=autotvm.LocalRunner(
                    number=10,
                    repeat=3,
                    timeout=100,
                    enable_cpu_cache_flush=True if target == "llvm" else False,
                ),
            )

            filename = "mm.json"
            if os.path.isfile(filename):
                os.remove(filename)

            tuner = autotvm.tuner.DropletTuner(task)

            search_time = time.time()
            tuner.tune(
                n_trial=trials,
                measure_option=measure_option,
                callbacks=[autotvm.callback.log_to_file(filename)],
            )
            search_time = time.time() - search_time
            d_time, d_config = get_best_time(filename)

            d_time = np.mean(np.array(d_time))
            t_ansor = np.mean(t_ansor)

            f = open(filename, "r")
            count = 0
            for l in f.readlines():
                count += 1

            print(
                "%d, %.4f, %.2f, %d, %.4f, %.2f"
                % (i, d_time, search_time, count, t_ansor, t_ansor / d_time)
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "python print_record_info.py -m 'ansor' -a x86 -l 'results/cpu_matmul.json' -i 3"
    )
    parser.add_argument(
        "-m", "--method", type=str, required=True, help="Options: ansor, droplet"
    )
    parser.add_argument(
        "-a", "--arch", type=str, required=True, help="Options: x86, arm, cuda"
    )
    parser.add_argument("-l", "--logfile", type=str, required=True)
    parser.add_argument("-i", "--index", type=int, default=-1)
    parser.add_argument("-t", "--trials", type=int, default=100)
    args = parser.parse_args()

    method = args.method
    arch = args.arch
    logfile = args.logfile
    index = args.index
    trials = args.trials

    if arch == "x86":
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

    if method == "ansor":
        generate_ansor_template(logfile, target, trials)
    elif method == "droplet":
        build_template(logfile, index, target, trials)
