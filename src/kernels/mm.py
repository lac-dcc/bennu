import os, sys
from tvm import te
from tvm import autotvm, auto_scheduler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.module.template_factory import Template_factory


@auto_scheduler.register_workload
def ansor_mm(N, L, M, dtype="float32"):
    A = te.placeholder((N, L), name="A", dtype=dtype)
    B = te.placeholder((L, M), name="B", dtype=dtype)

    k = te.reduce_axis((0, L), name="k")
    C = te.compute((N, M), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")

    return [A, B, C]


@autotvm.template("autotvm_mm")
def autotvm_mm(N, K, M, dtype="float32", cfg=None):
    k = te.reduce_axis((0, K), "k")
    A = te.placeholder((M, K), name="A")
    B = te.placeholder((K, N), name="B")
    C = te.compute((M, N), lambda m, n: te.sum(A[m, k] * B[k, n], axis=k), name="C")

    args = [A, B, C]
    tensors = C

    if cfg is not None:
        return Template_factory(cfg, tensors, args)
    else:
        return te.create_schedule(C.op), args
