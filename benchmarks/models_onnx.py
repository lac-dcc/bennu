import tvm, argparse, os, sys, json, time
from tvm import auto_scheduler
from tvm.driver import tvmc
from tvm.driver.tvmc.autotuner import autoscheduler_get_tuning_tasks
from tvm.auto_scheduler import MeasureInput, MeasureResult

from tvm.auto_scheduler.search_task import SearchTask

import tvm._ffi
from tvm.auto_scheduler import _ffi_api

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.utils import *
from src.DropletSearch import Droplet

num_threads = os.cpu_count()
os.environ["TVM_NUM_THREADS"] = str(num_threads)


def generate_ansor_template(bench, logfile, target, trials):
    model = tvmc.load(bench)
    clean_file(logfile)
    start = time.time()
    tvmc.tune(
        tvmc_model=model,
        target=target,
        tuning_records=logfile,
        repeat=3,
        timeout=100,
        parallel=os.cpu_count(),
        trials=trials,
        enable_autoscheduler=True,
    )
    end = time.time()
    print("time search:", end - start)


def build_template(bench, logfile, index, target, trials, top=1000):
    model = tvmc.load(bench)
    tasks, weights = autoscheduler_get_tuning_tasks(
        mod=model.mod, params=model.params, target=target
    )

    if "ansor" in logfile:
        droplet_log = logfile.replace("ansor", "droplet")
    else:
        droplet_log = ".".join(logfile.split(".")[:-1]) + "_droplet.json"
    clean_file(droplet_log)

    cfg = get_best_multilayers(logfile, top)
    cfg_10k = get_best_multilayers(logfile, 10000)

    print(
        "Layer, Time Droplet (s), Tuning time Droplet (s), tasks Droplet, Time Ansor (s), tasks Ansor, speedup"
    )

    for layer, workload in enumerate(cfg):
        if index != -1 and layer != index:
            continue

        log = f"layer_{layer}.log"
        clean_file(log)

        _, _, json_file = cfg[workload]
        t, _, _ = cfg_10k[workload]  # get the best value in 10k
        droplet = Droplet(json_file, workload, target, log, trials)
        start = time.time()
        droplet.tune()
        end = time.time()

        droplet_avg, droplet_cfg = get_best_time(log)

        print(
            "%d, %.8f, %.2f, %d, %.8f, %d, %.2f"
            % (
                layer,
                np.mean(droplet_avg),
                end - start,
                get_tasks(log),
                np.mean(t),
                min(top, get_task_multilayers(logfile)[workload]),
                np.mean(t) / np.mean(droplet_avg),
            )
        )
        append_file(droplet_cfg, droplet_log)


def run(logfile, bench, target, dev):
    model = tvmc.load(bench)

    package = tvmc.compile(model, target=target, tuning_records=logfile)

    package = tvmc.compile(model, target=target)

    result = tvmc.run(package, device=dev)
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "python print_record_info.py -m ansor -a x86 -l results/model.json -i 3"
    )
    parser.add_argument(
        "-m", "--method", type=str, required=True, help="Options: ansor, droplet"
    )
    parser.add_argument(
        "-a", "--arch", type=str, required=True, help="Options: x86, arm, cuda"
    )
    parser.add_argument("-l", "--logfile", type=str, required=True)
    parser.add_argument("-b", "--benchmark", type=str, required=True)
    parser.add_argument("-i", "--index", type=int, default=-1)
    parser.add_argument("-t", "--trials", type=int, default=100)
    parser.add_argument("-k", "--top", type=int, default=1000)
    args = parser.parse_args()

    method = args.method
    arch = args.arch
    logfile = args.logfile
    bench = args.benchmark
    index = args.index
    trials = args.trials
    top = args.top

    if arch == "x86":
        target_name = "llvm"
        target = tvm.target.Target("llvm")
        dev = tvm.cpu()
    elif arch == "cuda":
        target_name = "cuda"
        target = tvm.target.Target("cuda")
        dev = tvm.cuda()
    elif arch == "arm":
        target_name = "llvm -mcpu=a64fx"
        target = tvm.target.Target("llvm -mcpu=a64fx")
        dev = tvm.cpu()
    else:
        print("Archtecture doesn't support.")
        exit(0)

    if method == "ansor":
        generate_ansor_template(bench, logfile, target_name, trials)
    elif method == "droplet":
        build_template(bench, logfile, index, target, trials, top)
    elif method == "run":
        run(logfile, target, dev)
