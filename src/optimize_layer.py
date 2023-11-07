import time, sys, os
from tvm import autotvm
from tvm.auto_scheduler.workload_registry import workload_key_to_tensors

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.template_factory import Template_factory
from src.utils import *


@autotvm.template("layer")
def layer(workload_key, cfg):
    tensors = workload_key_to_tensors(workload_key)
    cfg = convert_to_list(cfg)
    return Template_factory(cfg, tensors)


def execute_one_layer(workload_key, cfg, target, trials):
    task = autotvm.task.create("layer", args=(workload_key, cfg), target=target)

    measure_option = autotvm.measure_option(
        builder="local",
        runner=autotvm.LocalRunner(
            number=10,
            repeat=3,
            timeout=100,
            enable_cpu_cache_flush=True if target == "llvm" else False,
        ),
    )

    filename = "layer.json"
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
        % (0, d_time, search_time, count, t_ansor, t_ansor / d_time)
    )
