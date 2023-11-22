import os, sys, time, argparse
from tvm import relay, auto_scheduler
from tvm.relay import testing

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.utils import *
from src.DropletSearch import Droplet

## ------------------ Global ---------------------
batch_size = 1
input_shape = (batch_size, 3, 224, 224)
output_shape = (batch_size, 1000)
layout = "NCHW"
dtype = "float32"


def resnet18_ansor(batch_size, target):
    n_layer = 18
    mod, params = relay.testing.resnet.get_workload(
        num_layers=n_layer, batch_size=batch_size, dtype=dtype, layout=layout
    )
    tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)

    return tasks, task_weights, mod, params


def generate_ansor_template(log_file, target, trials):
    tasks, task_weights, mod, params = resnet18_ansor(batch_size, target)

    ## Set Parameters for Auto-Scheduler
    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=trials,  # change this to 20000 to achieve the best performance
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
    tuner.tune(tune_option)
    end = time.time()

    # compile kernels in kernel tuned only mode
    """
    with auto_scheduler.ApplyHistoryBest(log_file):
        with tvm.transform.PassContext(
            opt_level=3, config={"relay.backend.use_auto_scheduler": True}
        ):
            lib = relay.build(mod, target=target, params=params)
            r = evaluate_performance(lib, input_shape, target)
    """

    print("Time for each layers")
    cfg = get_best_multilayers(logfile)
    for i, v in enumerate(cfg):
        print(f"Layer {i}: Time {np.mean(cfg[v][0])}")
    # print("\nTime to execute the algorithm: ", np.mean(r))
    print("Time spent to search:", end - start)


def build_template_multilayers(logfile, target, trials):
    tasks, task_weights, mod, params = resnet18_ansor(batch_size, target)
    cfg = get_best_multilayers(logfile)

    for task in tasks:
        auto_scheduler.workload_registry.register_workload_tensors(
            task.workload_key, task.compute_dag.tensors
        )

    print(
        "Layer, Time Droplet (s), Tuning time Droplet (s), tasks Droplet, Time Ansor (s), tasks Ansor, speedup"
    )
    for layer, workload in enumerate(cfg):
        log = f"layer_{layer}.log"
        clean_file(log)

        t, _, json_file = cfg[workload]
        droplet = Droplet(json_file, workload, target, log, trials)
        start = time.time()
        droplet.tune()
        end = time.time()

        droplet_avg, droplet_cfg = get_best_time(log)

        print(
            "%d, %.7f, %.2f, %d, %.7f, %d, %.2f"
            % (
                layer,
                np.mean(droplet_avg),
                end - start,
                get_tasks(log),
                np.mean(t),
                get_task_multilayers(logfile)[workload],
                np.mean(t) / np.mean(droplet_avg),
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "python print_record_info.py -m 'ansor' -a x86 -l 'results/cpu_matmul.json' -i 3"
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
        build_template_multilayers(logfile, target, trials)
