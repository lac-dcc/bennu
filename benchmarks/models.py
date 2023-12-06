import os, sys, time, argparse
from tvm import relay, auto_scheduler
from tvm.relay import testing

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.utils import *
from src.DropletSearch import Droplet

def get_network(name, batch_size, layout="NHWC", dtype="float32"):
    """Get the symbol definition and random weight of a network"""

    # auto-scheduler prefers NHWC layout
    if layout == "NHWC":
        image_shape = (224, 224, 3)
    elif layout == "NCHW":
        image_shape = (3, 224, 224)
    else:
        raise ValueError("Invalid layout: " + layout)

    input_shape = (batch_size,) + image_shape
    output_shape = (batch_size, 1000)

    if name.startswith("resnet-"):
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.resnet.get_workload(
            num_layers=n_layer,
            batch_size=batch_size,
            layout=layout,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name.startswith("resnet3d-"):
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.resnet.get_workload(
            num_layers=n_layer,
            batch_size=batch_size,
            layout=layout,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name == "mobilenet":
        mod, params = relay.testing.mobilenet.get_workload(
            batch_size=batch_size, layout=layout, dtype=dtype, image_shape=image_shape
        )
    elif name == "squeezenet_v1.1":
        assert layout == "NCHW", "squeezenet_v1.1 only supports NCHW layout"
        mod, params = relay.testing.squeezenet.get_workload(
            version="1.1",
            batch_size=batch_size,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name == "inception_v3":
        input_shape = (batch_size, 3, 299, 299) if layout == "NCHW" else (batch_size, 299, 299, 3)
        mod, params = relay.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == "mxnet":
        # an example for mxnet model
        from mxnet.gluon.model_zoo.vision import get_model

        assert layout == "NCHW"

        block = get_model("resnet50_v1", pretrained=True)
        mod, params = relay.frontend.from_mxnet(block, shape={"data": input_shape}, dtype=dtype)
        net = mod["main"]
        net = relay.Function(
            net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs
        )
        mod = tvm.IRModule.from_expr(net)
    elif name == 'bert':
        import torch
        import transformers  # pip3 install transfomers==3.0
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        model_class = transformers.BertModel
        tokenizer_class = transformers.BertTokenizer

        # You can also download them manualy
        #   https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin
        #   https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt
        #   https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json
        # Then rename to pytorch_model.bin, vocab.txt & config.json
        # weight = 'path to downloaded model dir'
        weight = 'bert-base-uncased'
        model = model_class.from_pretrained(weight)
        model.eval()

        # tokenizer = tokenizer_class.from_pretrained(weight)
        # A = torch.tensor([tokenizer.encode("Here is some text to encode", add_special_tokens=True)])
        # There is 30522 words in bert-base-uncased's vocabulary list
        input_shape = [batch_size, 128]
        input_name = 'input_ids'
        input_dtype = 'int64'
        A = torch.randint(30000, input_shape)
        scripted_model = torch.jit.trace(model, [A], strict=False)
        shape_list = [('input_ids', input_shape)]
        mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

        mod = tvm.relay.transform.FastMath()(mod)
        mod = tvm.relay.transform.EliminateCommonSubexpr()(mod)
        BindPass = tvm.relay.transform.function_pass(lambda fn, new_mod, ctx:
                            tvm.relay.build_module.bind_params_by_name(fn, params), opt_level=1)
        mod = BindPass(mod)
        mod = tvm.relay.transform.FoldConstant()(mod)
        mod = tvm.relay.transform.CombineParallelBatchMatmul()(mod)
        mod = tvm.relay.transform.FoldConstant()(mod)
    else:
        raise f"Error: not find {name} model"

    return mod, params, input_shape, output_shape


def generate_ansor_template(name, log_file, target, trials):
    batch_size = 1
    mod, params, input_shape, output_shape = get_network(name, batch_size)

    if os.path.exists(log_file):
        os.remove(log_file)
    tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)

    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=trials,  # change this to 20000 to achieve the best performance
        runner=auto_scheduler.LocalRunner(number=2, repeat=3, min_repeat_ms=100, enable_cpu_cache_flush=True if target=="llvm" else False),
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=0
    )
    start = time.time()
    tuner.tune(tune_option)
    end = time.time()
    print("Time search: %.2f" %(end-start))

    # compile kernels in kernel tuned only mode
    """
    compile kernels in kernel tuned only mode
        with auto_scheduler.ApplyHistoryBest(log_file):
            with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
                lib = relay.build(mod, target=target, params=params)
                if model != "bert":
                    r = evaluate_performance(lib, input_shape, target)
                else:
                    r = evaluate_performance(lib, input_shape, target, input_name="input_ids", dtype="int64")
    """

    print("Time for each layers")
    cfg = get_best_multilayers(logfile)
    for i, v in enumerate(cfg):
        print(f"Layer {i}: Time {np.mean(cfg[v][0])}")
    # print("\nTime to execute the algorithm: ", np.mean(r))
    print("Time spent to search:", end - start)


def build_template_multilayers(name, logfile, target, trials):
    batch_size = 1
    mod, params, input_shape, output_shape = get_network(name, batch_size)
    tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)

    cfg = get_best_multilayers(logfile)

    #for task in tasks:
    #    auto_scheduler.workload_registry.register_workload_tensors(
    #        task.workload_key, task.compute_dag.tensors
    #    )

    print(
        "Layer, Time Droplet (s), Tuning time Droplet (s), tasks Droplet, Time Ansor (s), tasks Ansor, speedup"
    )
    for layer, workload in enumerate(cfg):
        log = f"layer_{layer}.log"
        clean_file(log)

        t, _, json_file = cfg[workload]
        droplet = Droplet(json_file, workload, target, log, trials)
        start = time.time()
        droplet.tune()
        end = time.time()

        droplet_avg, droplet_cfg = get_best_time(log)

        print(
            "%d, %.7f, %.2f, %d, %.7f, %d, %.2f"
            % (
                layer,
                np.mean(droplet_avg),
                end - start,
                get_tasks(log),
                np.mean(t),
                get_task_multilayers(logfile)[workload],
                np.mean(t) / np.mean(droplet_avg),
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "python print_record_info.py -m 'ansor' -a x86 -l 'results/cpu_matmul.json' -i 3"
    )
    parser.add_argument(
        "-m", "--method", type=str, required=True, help="Options: ansor, droplet"
    )
    parser.add_argument(
        "-a", "--arch", type=str, required=True, help="Options: x86, aarch64, cuda"
    )
    parser.add_argument("-l", "--logfile", type=str, required=True)
    parser.add_argument("-b", "--benchmark", type=str, required=True)
    parser.add_argument("-t", "--trials", type=int, default=100)
    args = parser.parse_args()

    method = args.method
    arch = args.arch
    bench = args.benchmark
    logfile = args.logfile
    trials = args.trials

    if arch == "x86":
        target = tvm.target.Target("llvm")
        dev = tvm.cpu()
    elif arch == "cuda":
        target = tvm.target.Target("cuda")
        dev = tvm.cuda()
    elif arch == "aarch64":
        target = tvm.target.Target("llvm -mcpu=a64fx")
        dev = tvm.cpu()
    else:
        print("Archtecture doesn't support.")
        exit(0)

    if method == "ansor":
        generate_ansor_template(bench, logfile, target, trials)
    elif method == "droplet":
        build_template_multilayers(bench, logfile, target, trials)
