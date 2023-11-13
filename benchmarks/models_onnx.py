import tvm, argparse
from copy import deepcopy
from tvm import auto_scheduler
from tvm.driver import tvmc
from tvm.driver.tvmc.autotuner import autoscheduler_get_tuning_tasks


def generate_ansor_template(bench, logfile, target, trials):
    model = tvmc.load(bench)
    tvmc.tune(
        tvmc_model=model,
        target="llvm",
        tuning_records=logfile,
        repeat=3,
        timeout=100,
        trials=trials,
        enable_autoscheduler=True
    )


def build_template(bench, logfile, index, target, trials):
    model = tvmc.load(bench)
    tasks, weights = autoscheduler_get_tuning_tasks(
        mod=deepcopy(model.mod), params=model.params, target=target
    )

    for task in tasks:
        auto_scheduler.workload_registry.register_workload_tensors(
            task.workload_key, task.compute_dag.tensors
        )
        # print(task.workload_key, task.compute_dag.tensors)

    #inputs, results = auto_scheduler.RecordReader(logfile).read_lines()
    # working in the logic.


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
