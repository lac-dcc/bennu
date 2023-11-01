import os, sys, time, argparse
from tvm import te, topi, relay, autotvm, auto_scheduler
from tvm.relay import testing

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.template_factory import Template_factory
from src.utils import *

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


def resnet18_autotvm(batch_size, target, cfg=None):
    """
    if cfg is None:
        tasks = autotvm.task.extract_from_program(
            mod["main"], target=target, params=params
        )
    else:
        tasks = []
        tasks.append(Template_factory(cfg, mod["main"], params))

    return tasks
    """
    pass


def generate_ansor_template(log_file, target):
    tasks, task_weights, mod, params = resnet18_ansor(batch_size, target)

    ## Set Parameters for Auto-Scheduler
    trial = 1000
    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
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
    tuner.tune(tune_option)
    end = time.time()

    # compile kernels in kernel tuned only mode
    with auto_scheduler.ApplyHistoryBest(log_file):
        with tvm.transform.PassContext(
            opt_level=3, config={"relay.backend.use_auto_scheduler": True}
        ):
            lib = relay.build(mod, target=target, params=params)
            r = evaluate_performance(lib, input_shape, target)

    print("Time for each layers")
    results = get_best_time_multilayer(log_file)
    for i, v in enumerate(results):
        print(f"Layer {i}: Time {np.mean(results[v][0])} cfg: {results[v][1]}")
    print("\nTime to execute the algorithm: ", np.mean(r))
    print("Time spent to search:", end - start)


def build_template(log_file, index, target):
    """
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
                runner=autotvm.LocalRunner(number=10, repeat=3, timeout=100, enable_cpu_cache_flush=True if target == "llvm" else False),
            )

            filename = "mm.json"
            if os.path.isfile(filename):
                os.remove(filename)

            tuner = autotvm.tuner.DropletTuner(task)

            search_time = time.time()
            tuner.tune(
                n_trial=100,
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
    """
    pass


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
    args = parser.parse_args()

    method = args.method
    arch = args.arch
    logfile = args.logfile
    index = args.index

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
        generate_ansor_template(logfile, target)
    elif method == "droplet":
        build_template(logfile, index, target)
