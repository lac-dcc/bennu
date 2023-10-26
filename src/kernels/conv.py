import os, sys
from tvm import te
from tvm import autotvm, auto_scheduler
import tvm.topi as topi

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.module.template_factory import Template_factory


@auto_scheduler.register_workload
def conv2d_ansor(
    input_shape,
    filter_shape,
    strides=(1, 1),
    padding=(1, 1),
    dilation=(1, 1),
    layout="NCHW",
    dtype="float32",
):
    batch, in_channel, in_height, in_width = input_shape
    num_filter, kernel_height, kernel_width = filter_shape

    A = te.placeholder((batch, in_channel, in_height, in_width), name="A", dtype=dtype)
    W = te.placeholder(
        (num_filter, in_channel, kernel_height, kernel_width), name="W", dtype=dtype
    )

    C = topi.nn.conv2d(
        A, W, strides, padding, dilation, data_layout=layout, out_dtype=dtype
    )

    return [A, W, C]


@autotvm.template("conv2d")
def conv2d(
    input_shape,
    filter_shape,
    strides=(1, 1),
    padding=(1, 1),
    dilation=(1, 1),
    layout="NCHW",
    dtype="float32",
    cfg=None,
):
    batch, in_channel, in_height, in_width = input_shape
    num_filter, kernel_height, kernel_width = filter_shape

    A = te.placeholder((batch, in_channel, in_height, in_width), name="A", dtype=dtype)
    W = te.placeholder(
        (num_filter, in_channel, kernel_height, kernel_width), name="W", dtype=dtype
    )

    C = topi.nn.conv2d(
        A, W, strides, padding, dilation, data_layout=layout, out_dtype=dtype
    )

    args = [A, W, C]
    tensors = C

    if cfg is not None:
        return Template_factory(cfg, tensors, args)
    else:
        return te.create_schedule(C.op), args