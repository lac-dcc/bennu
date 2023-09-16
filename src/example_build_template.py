import logging
import sys, os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
import tvm
from tvm import te
import tvm.testing

from tvm import autotvm
from src.module.utils import get_best_time
from src.module.creating_template import Template_autotvm
from src.module.template_factory import Template_factory
from src.kernels.mm import autotvm_mm

if __name__ == "__main__":

    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    else:
        print("Example to execute:")
        print("python3 src/example_build_template.py <file>.json")
        exit(1)

    best_time, best_config = get_best_time(json_file)
    
    N, L, M = 1000, 800, 700
    task = autotvm.task.create("autotvm_mm", args=(N, L, M, "float32", best_config), target="llvm")
    print(task.config_space)

    logging.getLogger("autotvm").setLevel(logging.DEBUG)
    logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))

    measure_option = autotvm.measure_option(
        builder="local", 
        runner=autotvm.LocalRunner(number=5, repeat=3)
    )

    filename = "matmul.json"
    if os.path.isfile(filename):
        os.remove(filename)

    tuner = autotvm.tuner.DropletTuner(task)
    tuner.tune(
        n_trial=1000,
        measure_option=measure_option,
        callbacks=[autotvm.callback.log_to_file(filename)],
    )

    # apply history best from log file
    '''
    with autotvm.apply_history_best(filename):
        with tvm.target.Target("llvm"):
            s, arg_bufs = matmul(N, L, M, "float32", best_config)
            func = tvm.build(s, arg_bufs)

    # check correctness
    a_np = np.random.uniform(size=(N, L)).astype(np.float32)
    b_np = np.random.uniform(size=(L, M)).astype(np.float32)
    c_np = a_np.dot(b_np)

    c_tvm = tvm.nd.empty(c_np.shape)
    func(tvm.nd.array(a_np), tvm.nd.array(b_np), c_tvm)

    tvm.testing.assert_allclose(c_np, c_tvm.numpy(), rtol=1e-4)
    '''

    best_time, best_config = get_best_time(filename)

    print(best_time, best_config)
