import logging
import sys, os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
import tvm
from tvm import te, topi
import tvm.testing

from tvm import autotvm
from src.module.utils import get_best_time
from src.module.creating_template import Template_autotvm
from src.kernels.conv import conv2d_autotvm

"""
Example 1:
    Time spent: [0.0173736, 0.0208746, 0.0257531]
    [["CHW", 3, "local"], 
    ["SP", 3, 0, 1, [1, 1, 1], 1], 
    ["SP", 3, 4, 64, [1, 32, 1], 1], 
    ["SP", 3, 8, 224, [7, 4, 8], 1], 
    ["SP", 3, 12, 224, [7, 2, 1], 1], 
    ["SP", 3, 16, 3, [1], 1], 
    ["SP", 3, 18, 3, [1], 1], 
    ["SP", 3, 20, 3, [3], 1], 
    ["RE", 3, [0, 4, 8, 12, 1, 5, 9, 13, 16, 18, 20, 2, 6, 10, 14, 17, 19, 21, 3, 7, 11, 15]], 
    ["FSP", 4, 0, 1, 2], 
    ["FSP", 4, 3, 2, 2], 
    ["FSP", 4, 6, 3, 2], 
    ["FSP", 4, 9, 4, 2], 
    ["RE", 4, [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]], 
    ["CA", 3, 4, 7], 
    ["CA", 1, 3, 6], 
    ["FU", 4, [0, 1, 2, 3, 4, 5, 6]], 
    ["AN", 4, 0, 3], 
    ["PR", 3, 0, "auto_unroll_max_step$512"], 
    ["AN", 1, 3, 2], 
    ["AN", 3, 21, 2], 
    ["AN", 4, 5, 2]]]], 
    "r": [[0.00124181, 0.00117129, 0.001106], 0, 4.13106, 1698752640] 
"""


def example1(ta):
    ta.CHW([3, "local"])
    ta.SP([3, 0, 1, [1, 1, 1], 1])
    ta.SP([3, 4, 64, [1, 32, 1], 1])
    ta.SP([3, 8, 224, [7, 4, 8], 1])
    ta.SP([3, 12, 224, [7, 2, 1], 1])
    ta.SP([3, 16, 3, [1], 1])
    ta.SP([3, 18, 3, [1], 1])
    ta.SP([3, 20, 3, [3], 1])
    ta.RE(
        [
            3,
            [
                0,
                4,
                8,
                12,
                1,
                5,
                9,
                13,
                16,
                18,
                20,
                2,
                6,
                10,
                14,
                17,
                19,
                21,
                3,
                7,
                11,
                15,
            ],
        ]
    )
    ta.FSP([4, 0, 1, 2])
    ta.FSP([4, 3, 2, 2])
    ta.FSP([4, 6, 3, 2])
    ta.FSP([4, 9, 4, 2])
    ta.RE([4, [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]])
    ta.CA([3, 4, 7])
    ta.CA([1, 3, 6])
    ta.FU([4, [0, 1, 2, 3, 4, 5, 6]])
    ta.AN([4, 0, 3])
    ta.PR([3, 0, "auto_unroll_max_step$512"])
    ta.AN([1, 3, 2])
    ta.AN([3, 21, 2])
    ta.AN([4, 5, 2])
    return ta.ret()


@autotvm.template("conv2d")
def conv2d(input_shape, filter_shape):
    strides = (1, 1)
    padding = (1, 1)
    dilation = (1, 1)
    layout = "NCHW"
    dtype = "float32"

    A = te.placeholder(shape=input_shape, name="A", dtype=dtype)
    W = te.placeholder(shape=filter_shape, name="W", dtype=dtype)
    C = topi.nn.conv2d(
        A, W, strides, padding, dilation, data_layout=layout, out_dtype=dtype
    )

    args = [A, W, C]
    tensors = C

    ta = Template_autotvm(tensors, args)
    example1(ta)

    return ta.ret()


if __name__ == "__main__":
    input_shape = (1, 3, 224, 224)
    filter_shape = (64, 3, 3, 3)

    task = autotvm.task.create(
        "conv2d", args=(input_shape, filter_shape), target="llvm"
    )

    print(task.config_space)

    logging.getLogger("autotvm").setLevel(logging.DEBUG)
    logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))

    measure_option = autotvm.measure_option(
        builder="local", runner=autotvm.LocalRunner(number=5, repeat=3)
    )

    filename = "test.log"
    if os.path.isfile(filename):
        os.remove(filename)

    tuner = autotvm.tuner.GridSearchTuner(task)
    tuner.tune(
        n_trial=10,
        measure_option=measure_option,
        callbacks=[autotvm.callback.log_to_file(filename)],
    )

    """
    # apply history best from log file
    with autotvm.apply_history_best(filename):
        with tvm.target.Target("llvm"):
            s, arg_bufs = matmul(N, L, M, "float32")
            func = tvm.build(s, arg_bufs)

    # check correctness
    a_np = np.random.uniform(size=(N, L)).astype(np.float32)
    b_np = np.random.uniform(size=(L, M)).astype(np.float32)
    c_np = a_np.dot(b_np)

    c_tvm = tvm.nd.empty(c_np.shape)
    func(tvm.nd.array(a_np), tvm.nd.array(b_np), c_tvm)

    tvm.testing.assert_allclose(c_np, c_tvm.numpy(), rtol=1e-4)
    """

    best_time, best_config = get_best_time(filename)

    print(best_time, best_config)
