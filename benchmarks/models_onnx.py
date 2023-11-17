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


def generate_ansor_template(bench, logfile, target, trials):
    model = tvmc.load(bench)
    tvmc.tune(
        tvmc_model=model,
        target=target,
        tuning_records=logfile,
        repeat=3,
        timeout=100,
        trials=trials,
        enable_autoscheduler=True,
    )


def build_template(bench, logfile, index, target, trials):
    model = tvmc.load(bench)
    tasks, weights = autoscheduler_get_tuning_tasks(
        mod=model.mod, params=model.params, target=target
    )

    cfg = get_best_multilayers(logfile)

    print("Time Droplet (s), Tuning time (s), Time Ansor (s), speedup")
    layer = 0
    for workload in cfg:
        log = f"layer_{layer}.log"
        t, params, json_file = cfg[workload]
        droplet = Droplet(json_file, workload, target)
        start = time.time()
        droplet.tune(log)
        end = time.time()
        layer += 1

        droplet_avg, droplet_cfg = get_best_time(log)

        print(
            "%.6f, %.2f, %.6f, %.2f"
            % (
                np.mean(droplet_avg),
                end - start,
                np.mean(t),
                np.mean(t) / np.mean(droplet_avg),
            )
        )
        print(droplet_cfg)

        break


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
    args = parser.parse_args()

    method = args.method
    arch = args.arch
    logfile = args.logfile
    bench = args.benchmark
    index = args.index
    trials = args.trials

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
        build_template(bench, logfile, index, target, trials)
