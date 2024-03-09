import os, sys, time, argparse
import tvm
from tvm import auto_scheduler, te
from tvm.relay import testing
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.utils.utils import *
from src.DropletSearch import Droplet

@auto_scheduler.register_workload
def matmul_layer(batch, in_dim, out_dim):
    data = te.placeholder((batch, in_dim), name='A', dtype="float32")
    weight = te.placeholder((in_dim, out_dim), name='B', dtype="float32")
    k = te.reduce_axis((0, in_dim), name="k")
    matmul = te.compute((batch, out_dim), lambda x, y: te.sum(data[x, k] * weight[k, y], axis=k))
    return [data, weight, matmul]

def method_droplet(logfile, bench, top, target):
    
    if bench == "matmul":
        batch, in_dim, out_dim = 16, 1024, 1024
        matmul_layer(batch, in_dim, out_dim)
        task = auto_scheduler.SearchTask(
            func=matmul_layer, args=(batch, in_dim, out_dim), target=target
        )
    
    cfg = get_best_multilayers(logfile, top)
    log = f"{logfile}_droplet.log"

    print("avg (ms), std (ms), time total (min)")
    for layer, workload in enumerate(cfg):
        _, _, json_file = cfg[workload]

        method = Droplet(json_file, target, log)
        method.tune()

        droplet_avg, _ = get_best_time(log)
        time_m, _ = get_time_total(log)

        print("%.8f, %.8f, %.2f"
            % (
                np.mean(droplet_avg) * 1000, # ms
                np.std(droplet_avg) * 1000, # ms
                time_m / 60.0
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
    parser.add_argument("-t", "--trials", type=int, default=100)
    parser.add_argument("-k", "--top", type=int, default=1000)
    args = parser.parse_args()

    arch = args.arch
    logfile = args.logfile
    bench = args.benchmark
    trials = args.trials
    top = args.top

    if arch != "cuda":
        raise Exception("No support for any architecture other than cuda.")
    else:
        target_name = "cuda"
        target = tvm.target.Target("cuda")
        dev = tvm.cuda()
    
    method_droplet(logfile, bench, top, target)