import os, sys, time, argparse
from tvm import te, topi, autotvm, auto_scheduler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.template_factory import Template_factory
from src.utils import *

## ------------------ Global ---------------------
# 128 84 83 83 5  5  2       SAME
# N   CI H  W  KH KW Strides Padding
input_shape = (128, 84, 83, 83)
filter_shape = (84, 1, 5, 5)
strides = (1, 1)
padding = (1, 1)
dilation = (1, 1)
layout = "NCHW"
dtype = "float32"


## ----------------- Benchmark -------------------
@auto_scheduler.register_workload
def ansor_depthwise(input_shape, filter_shape, dtype="float32"):
    A = te.placeholder(input_shape, name="A", dtype=dtype)
    B = te.placeholder(filter_shape, name="B", dtype=dtype)
    C = topi.nn.depthwise_conv2d_nhwc(
        A, B, stride=strides, padding=padding, dilation=dilation, out_dtype=dtype
    )

    return [A, B, C]


@autotvm.template("autotvm_depthwise")
def autotvm_depthwise(input_shape, filter_shape, dtype, cfg=None):
    A = te.placeholder(input_shape, name="A", dtype=dtype)
    B = te.placeholder(filter_shape, name="B", dtype=dtype)
    C = topi.nn.depthwise_conv2d_nhwc(
        A, B, stride=strides, padding=padding, dilation=dilation, out_dtype=dtype
    )

    if cfg is not None:
        return Template_factory(cfg, [A, B, C])
    else:
        return te.create_schedule(C.op), [A, B, C]


## ---------------------------------------------


def generate_ansor_template(log_file, target, trials):
    task = tvm.auto_scheduler.SearchTask(
        func=ansor_depthwise, args=(input_shape, filter_shape, "float32"), target=target
    )

    ## Set Parameters for Auto-Scheduler
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=trials,  # change this to 20000 to achieve the best performance
        runner=auto_scheduler.LocalRunner(
            number=10,
            repeat=3,
            timeout=100,
            enable_cpu_cache_flush=True if target == "llvm -mcpu=a64fx" else False,
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
                "autotvm_depthwise",
                args=(input_shape, filter_shape, "float32", cfg_ansor),
                target=target,
            )

            measure_option = autotvm.measure_option(
                builder="local",
                runner=autotvm.LocalRunner(
                    number=10,
                    repeat=3,
                    timeout=100,
                    enable_cpu_cache_flush=True
                    if target == "llvm -mcpu=a64fx"
                    else False,
                ),
            )

            filename = "depth.json"
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
