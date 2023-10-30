import logging
import sys, os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
import tvm, time
from tvm import te
import tvm.testing

from tvm import autotvm
from src.module.utils import *
from src.kernels.mm import *
from src.kernels.conv import *
from src.kernels.resnet18 import *

if __name__ == "__main__":
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
        algorithm = sys.argv[2]
    else:
        print("Example to execute:")
        print("python3 src/example_build_template.py <file>.json <algorithm_name>")
        exit(1)

    config = get_template_ansor(json_file)

    print(
        "N, Droplet Time (s), Search Time (s), Count Sample, Ansor Time (s), speedup (Ansor/Droplet)"
    )
    c = 1
    for t_ansor, cfg_ansor in config:
        N, L, M = 1000, 800, 700
        task = autotvm.task.create(
            algorithm, args=(N, L, M, "float32", cfg_ansor), target="llvm"
        )

        # logging.getLogger("autotvm").setLevel(logging.DEBUG)
        # logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))

        measure_option = autotvm.measure_option(
            builder="local",
            runner=autotvm.LocalRunner(number=10, repeat=3),
        )

        filename = algorithm + ".json"
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
            % (c, d_time, search_time, count, t_ansor, t_ansor / d_time)
        )
        c += 1
