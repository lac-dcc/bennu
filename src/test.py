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
from src.kernels.mm import autotvm_mm


@autotvm.template("matmul")  # 1. use a decorator
def matmul(N, L, M, dtype):
    A = te.placeholder((N, L), name="A", dtype=dtype)
    B = te.placeholder((L, M), name="B", dtype=dtype)

    k = te.reduce_axis((0, L), name="k")
    C = te.compute((N, M), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")
    
    s = te.create_schedule(C.op)
    args = [A, B, C]
    tensors = C

    '''
        Example 1:
            Time spent: [0.0173736, 0.0208746, 0.0257531]
            Config: [[], 
            [['SP', 2, 0, 1000, [5, 25, 4], 1], ['SP', 2, 4, 700, [1, 35, 4], 1], ['SP', 2, 8, 800, [8], 1], 
            ['RE', 2, [0, 4, 1, 5, 8, 2, 6, 9, 3, 7]], 
            ['FU', 2, [0, 1, 2]], 
            ['AN', 2, 0, 3], 
            ['PR', 2, 0, 'auto_unroll_max_step$512'], 
            ['AN', 2, 7, 2]]]  
        Example 2:
            Time spent: [0.020076, 0.0222354, 0.0232053]
            Config: [[], 
            [['CHW', 2, 'local'], ['SP', 2, 0, 1000, [1, 20, 2], 1], ['SP', 2, 4, 700, [1, 7, 20], 1], ['SP', 2, 8, 800, [10], 1], 
            ['RE', 2, [0, 4, 1, 5, 8, 2, 6, 9, 3, 7]], 
            ['FSP', 3, 0, 1, 1], 
            ['FSP', 3, 2, 2, 1], 
            ['RE', 3, [0, 2, 1, 3]], 
            ['CA', 2, 3, 1], 
            ['AN', 3, 0, 3], 
            ['PR', 2, 0, 'auto_unroll_max_step$512']]]  
    '''
    
    ta = Template_autotvm(tensors, args)
    ta.CHW([2, 'local'])
    ta.SP([3, 3, 1])
    ta.RE_fixed([0, 4, 1, 5, 8, 2, 6, 9, 3, 7])
    #ta.FU_fixed([0, 1, 2])
    #ta.print()
    #ta.FSP_fixed([3, 0, 1, 1])
    #ta.FSP_fixed([3, 2, 2, 1])
    #ta.RE_fixed([0, 2, 1, 3])
    #ta.CA_fixed([2, 3, 1])
    #ta.PR_fixed([0, 'auto_unroll_max_step', 512])
    
    #ta.RE()
    #ta.print()

    return ta.ret()
    
    '''
    bn = 32
    kfactor = 4

    s = te.create_schedule(C.op)

    # Allocate write cache
    CC = s.cache_write(C, "local")

    mo, no, mi, ni = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)

    # Write cache is computed at no
    s[CC].compute_at(s[C], no)

    # New inner axes
    mc, nc = s[CC].op.axis

    (kaxis,) = s[CC].op.reduce_axis
    ko, ki = s[CC].split(kaxis, factor=kfactor)
    s[CC].reorder(ko, mc, ki, nc)
    #s[CC].vectorize(nc)

    # TODO: Add separate optimization step to discuss loop unrolling
    # unrolling is a loop optimization strategy which can reduce branch
    # prediction failures and increases the chance of concurrent execution
    # unroll kfactor loops
    #s[CC].unroll(ki)
    '''

    return s, args

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

    filename = "matmul.log"
    if os.path.isfile(filename):
        os.remove(filename)

    tuner = autotvm.tuner.GridSearchTuner(task)
    tuner.tune(
        n_trial=10,
        measure_option=measure_option,
        callbacks=[autotvm.callback.log_to_file(filename)],
    )

    '''
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
    '''

    best_time, best_config = get_best_time(filename)

    print(best_time, best_config)