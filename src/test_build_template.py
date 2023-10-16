import logging
import sys, os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
import tvm
from tvm import te
import tvm.testing

from tvm import autotvm
from src.module.utils import *
from src.kernels.mm import autotvm_mm

if __name__ == "__main__":

    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    else:
        print("Example to execute:")
        print("python3 src/example_build_template.py <file>.json")
        exit(1)

    config = get_template_ansor(json_file)

    for t_ansor, cfg_ansor in config:
        
        N, L, M = 1000, 800, 700
        task = autotvm.task.create("autotvm_mm", args=(N, L, M, "float32", cfg_ansor), target="llvm")

        #logging.getLogger("autotvm").setLevel(logging.DEBUG)
        #logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))

        try:
            measure_option = autotvm.measure_option(
                builder="local", 
                runner=autotvm.LocalRunner(number=5, repeat=3),
            )

            filename = "matmul.json"
            if os.path.isfile(filename):
                os.remove(filename)
        
            tuner = autotvm.tuner.DropletTuner(task)
            tuner.tune(
                n_trial=100,
                measure_option=measure_option,
                callbacks=[autotvm.callback.log_to_file(filename)],
            )
            d_time, d_config = get_best_time(filename)
        except:
            d_time = [10000]
        
        d_time = np.mean(np.array(d_time))
        t_ansor = np.mean(t_ansor)

        print("DROPLET: %.4f ANSOR: %.4f speed up (ANSOR/DROPLET %.2f)" %(d_time, t_ansor, t_ansor/d_time))