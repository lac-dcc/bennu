import logging
import sys

import numpy as np
import tvm
from tvm import te
import tvm.testing

from tvm import autotvm
from utils import get_best_time
from module.creating_template import Template_ansor


@autotvm.template("matmul")  # 1. use a decorator
def matmul(N, L, M, dtype):
    A = te.placeholder((N, L), name="A", dtype=dtype)
    B = te.placeholder((L, M), name="B", dtype=dtype)

    k = te.reduce_axis((0, L), name="k")
    C = te.compute((N, M), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")
    s = te.create_schedule(C.op)

    cfg = autotvm.get_config()
    args = [A, B, C]
    tensors = [C]

    # Config: [[], 
    #   [['CHW', 2, 'local'], 
    #   ['SP', 2, 0, 1000, [20, 1, 2], 1], 
    #   ['SP', 2, 4, 700, [1, 700, 1], 1], 
    #   ['SP', 2, 8, 800, [5], 1], 
    #   ['RE', 2, [0, 4, 1, 5, 8, 2, 6, 9, 3, 7]], 
    #   ['FSP', 3, 0, 1, 2], 
    #   ['FSP', 3, 3, 2, 2], 
    #   ['RE', 3, [0, 3, 1, 4, 2, 5]], 
    #   ['CA', 2, 3, 3], 
    #   ['FU', 3, [0, 1, 2]], 
    #   ['AN', 3, 0, 3], 
    #   ['PR', 2, 0, 'auto_unroll_max_step$512'], 
    #   ['AN', 2, 9, 2]]]

    bn = 32
    #kfactor = 4
    # Blocking by loop tiling
    #mo, no, mi, ni = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
    #(kaxis,) = s[C].op.reduce_axis
    #ko, ki = s[C].split(kaxis, factor=kfactor)

    # Hoist reduction domain outside the blocking loop
    #s[C].reorder(mo, no, ko, ki, mi, ni)

    #cfg.define_knob(["CHW", 0, 1])
    #if cfg["CHW"].val != 0:
    #    C = s.cache_write(C, "local")
    #print(tvm.lower(s, args))

    #mo, no, mi, ni = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)

    # Write cache is computed at no
    #s[CC].compute_at(s[C], no)

    ta = Template_ansor(s, tensors, cfg, args)
    ta.space(["CHW", 0, 'local']) # Error!
    ta.space(["SP", 0, 0, 1000, [20, 1, 20], 0])
    ta.space(['SP', 0, 1, 700, [1, 700, 1], 1])
    ta.space(['SP', 0, 2, 800, [5], 1])

    #ta.print()

    # schedule
    #y, x = s[C].op.axis
    #k = s[C].op.reduce_axis[0]

    # 3. define search space
    #cfg.define_knob("tile_y", [1, 2, 4, 8, 16])
    #cfg.define_knob("tile_x", [1, 2, 4, 8, 16])

    # 4. schedule according to config
    #x0, y0 = s[C].split(y, 20)
    #x1, y1 = s[C].split(y0, 1)
    #x2, y2 = s[C].split(y1, 2)
    #xo, xi = s[C].split(x, cfg["tile_x"].val)

    #print(tvm.lower(s, args))

    #s[C].reorder(yo, xo, k, yi, xi)

    return ta.ret()
    #return s, args

if __name__ == "__main__":
    N, L, M = 1000, 800, 700
    task = autotvm.task.create("matmul", args=(N, L, M, "float32"), target="llvm")
    print(task.config_space)

    logging.getLogger("autotvm").setLevel(logging.DEBUG)
    logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))

    measure_option = autotvm.measure_option(
        builder="local", 
        runner=autotvm.LocalRunner(number=5, repeat=3)
    )

    tuner = autotvm.tuner.RandomTuner(task)
    tuner.tune(
        n_trial=100,
        measure_option=measure_option,
        callbacks=[autotvm.callback.log_to_file("matmul.log")],
    )

    # apply history best from log file
    with autotvm.apply_history_best("matmul.log"):
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

    best_time, best_config = get_best_time("matmul.log")

    print(best_time, best_config)