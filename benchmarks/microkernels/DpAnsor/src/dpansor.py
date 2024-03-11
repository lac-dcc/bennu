import os, sys, time, argparse
import tvm
from tvm import auto_scheduler, te, topi
from tvm.topi.nn.utils import get_pad_tuple
from tvm.relay import testing
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.utils import *
from DropletSearch import Droplet

@auto_scheduler.register_workload
def matmul_layer(batch, in_dim, out_dim):
    data = te.placeholder((batch, in_dim), name="A", dtype="float32")
    weight = te.placeholder((in_dim, out_dim), name="B", dtype="float32")
    k = te.reduce_axis((0, in_dim), name="k")
    matmul = te.compute(
        (batch, out_dim), lambda x, y: te.sum(data[x, k] * weight[k, y], axis=k)
    )
    return [data, weight, matmul]

@auto_scheduler.register_workload
def conv2d_layer(N, CI, H, W, CO, KH, KW, strides, padding):
    data = te.placeholder((N, CI, H, W), name="input0")
    kernel = te.placeholder((CO, CI, KH, KW), name="input1")
    C = topi.nn.conv2d_nchw(
        data, kernel, strides, padding, dilation=1, out_dtype="float32"
    )
    return [data, kernel, C]

@auto_scheduler.register_workload
def depthwise_conv2d_layer(N, CI, H, W, KH, KW, strides, padding):
    input_shape, filter_shape= (N, CI, H, W), (CI, 1, KH, KW)
    data = te.placeholder(input_shape, name='data', dtype="float32")
    kernel = te.placeholder(filter_shape, name='kernel', dtype="float32")
    depthwise_conv = topi.nn.depthwise_conv2d_nchw(data, kernel, strides, padding, dilation=1, out_dtype="float32")
    return [data, kernel, depthwise_conv]

@auto_scheduler.register_workload
def pool_layer(pool_type, N, CI, H, W, KH, KW, strides, padding):
    data = te.placeholder((N, CI, H, W), name="data")
    pool = topi.nn.pool2d(
        data,
        (KH, KW),
        (strides, strides),
        (1, 1),
        get_pad_tuple(padding, (KH, KW)),
        pool_type=pool_type,
    )
    return [data, pool]

@auto_scheduler.register_workload
def reduction_layer(shape, axis, keep_dim):
    data = te.placeholder(shape, name="data")
    reduction = topi.sum(data, axis, keep_dim)
    return [data, reduction]

@auto_scheduler.register_workload
def relu_layer(*shape):
    data = te.placeholder(shape, name="data")
    relu = topi.nn.relu(data)
    return [data, relu]

def microkernels(bench):

    if bench == "matmul":
        batch, in_dim, out_dim = 1, 1024, 1024
        matmul_layer(batch, in_dim, out_dim)
        task = auto_scheduler.SearchTask(
            func=matmul_layer, args=(batch, in_dim, out_dim), target=target
        )
    elif bench == "conv2d":
        N, CI, H, W, CO, KH, KW, strides, padding = 128, 128, 28, 28, 128, 3, 3, 1, "SAME"
        task = auto_scheduler.SearchTask(
            func=conv2d_layer, args=(N, CI, H, W, CO, KH, KW, strides, padding), target=target,
        )
    elif bench == "depthwise":
        N, CI, H, W, KH, KW, strides, padding = 128, 84, 83, 83, 5, 5, 2, "SAME"
        task = auto_scheduler.SearchTask(
            func=depthwise_conv2d_layer, args=(N, CI, H, W, KH, KW, strides, padding), target=target
        )
    elif bench == "pooling":
        pool_type, N, CI, H, W, KH, KW, strides, padding = "avg", 128, 168, 83, 83, 1, 1, 2, "VALID"
        task = auto_scheduler.SearchTask(
            func=pool_layer, args=(pool_type, N, CI, H, W, KH, KW, strides, padding), target=target,
        )
    elif bench == "reduce":
        shape, axis, keep_dim = (128, 512, 1024), 2, False
        task = auto_scheduler.SearchTask(
            func=reduction_layer, args=(shape, axis, keep_dim), target=target
        )
    elif bench == "relu":
        shape = (4096, 4096)
        task = auto_scheduler.SearchTask(func=relu_layer, args=shape, target=target)
    else:
        raise Exception(
            f"Bench {bench} is not implemented! The benchmarks that microkernel give support are: \n -> matmul\n -> depthwise\n -> pooling\n -> reduce\n -> relu"
        )
    return task

def ansor(logfile, bench, target, trial=1000):

    task = microkernels(bench)

    # clean the log
    clean_file(logfile)

    measure_ctx = auto_scheduler.LocalRPCMeasureContext(
        number=3, repeat=1, timeout=20, min_repeat_ms=100
    )

    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=trial,  # top-k
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(logfile)],
        verbose=0,
    )

    # Run Ansor
    task.tune(tune_option)

def dpansor(logfile, bench, target, top=1000):

    task = microkernels(bench)

    # DPAnsor
    cfg = get_best_multilayers(logfile, top)
    log = f"{logfile}_droplet.log"

    print("results: ")
    for layer, workload in enumerate(cfg):
        _, _, json_file = cfg[workload]

        # DP Ansor call method
        method = Droplet(json_file, target, log)
        method.tune()

        droplet_avg, _ = get_best_time(log)
        droplet_time, droplet_trial = get_time_total(log)
        ansor_time, _ = get_time_total(logfile, top)

        ansor_avg_1000, _ = get_best_time(logfile)
        ansor_time_1000, _ = get_time_total(logfile)

        print(
            "%s, %.8f, %.8f, %d, %.2f, %.8f, %.8f, %.2f"
            % (
                bench,
                np.mean(droplet_avg) * 1000,  # ms
                np.std(droplet_avg) * 1000,  # ms
                droplet_trial,
                (droplet_time + ansor_time) / 60.0, # min
                np.mean(ansor_avg_1000) * 1000,
                ansor_time_1000,
                np.mean(ansor_avg_1000) / np.mean(droplet_avg)  
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "python dpansor.py -a cuda -l results/model.json -i 3"
    )
    parser.add_argument(
        "-a", "--arch", type=str, required=True, help="Options: x86, arm, cuda"
    )
    parser.add_argument("-l", "--logfile", type=str, required=True)
    parser.add_argument("-b", "--benchmark", type=str, required=True)
    parser.add_argument("-m", "--method", type=str, required=True)
    parser.add_argument("-k", "--top", type=int, default=100)
    parser.add_argument("-t", "--trials", type=int, default=1000)
    args = parser.parse_args()

    arch = args.arch
    logfile = args.logfile
    bench = args.benchmark
    method = args.method
    trials = args.trials
    top = args.top

    if arch != "cuda":
        raise Exception("No support for any architecture other than cuda.")
    else:
        target_name = "cuda"
        target = tvm.target.Target("cuda")
        dev = tvm.cuda()

    if method == "ansor":
        ansor(logfile, bench, target, trials)
    elif method == "dpansor":
        dpansor(logfile, bench, target, top)
