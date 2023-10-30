import os, sys
from tvm import relay, autotvm, auto_scheduler
from tvm.relay import testing

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.module.template_factory import Template_factory


def resnet18(batch_size, layout="NCHW", dtype="float32"):
    input_shape = (batch_size, 3, 224, 224)
    output_shape = (batch_size, 1000)

    n_layer = 18
    mod, params = relay.testing.resnet.get_workload(
        num_layers=n_layer, batch_size=batch_size, dtype=dtype, layout=layout
    )

    return mod, params, input_shape, output_shape


def resnet18_ansor(batch_size, target):
    mod, params, data_shape, out_shape = resnet18(batch_size)
    tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)

    return tasks, task_weights, mod, params, data_shape, out_shape


def resnet18_autotvm(batch_size, target, cfg=None):
    mod, params, data_shape, out_shape = resnet18(batch_size)

    if cfg is None:
        tasks = autotvm.task.extract_from_program(
            mod["main"], target=target, params=params
        )
    else:
        tasks = []
        tasks.append(Template_factory(cfg, mod["main"], params))

    return tasks
