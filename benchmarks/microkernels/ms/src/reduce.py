import os, sys, time, argparse, tvm
from tvm import te, topi
from tvm import meta_schedule as ms
from tvm.meta_schedule.runner.config import EvaluatorConfig
from tvm.script import tir as T

num_threads = os.cpu_count()
os.environ["TVM_NUM_THREADS"] = str(num_threads)
os.environ["MKL_NUM_THREADS"] = str(num_threads * 2 // 3)
os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads * 2 // 3)
os.environ["OMP_NUM_THREADS"] = str(num_threads * 2 // 3)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.utils import *

## ------------------ Global ---------------------
shape, axis, keep_dim = (128, 512, 1024), 2, False
dtype = "float32"

## ----------------- Benchmark -------------------
def print_reduce(shape, axis, keep_dim):
    data = te.placeholder(shape, name="data")
    reduction = topi.sum(data, axis, keep_dim)
    te.create_prim_func([data, reduction]).show()


@tvm.script.ir_module
class Main:
    @T.prim_func
    def main(data: T.Buffer((128, 512, 1024), "float32"), data_red: T.Buffer((128, 512), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, k2 in T.grid(128, 512, 1024):
            with T.block("data_red"):
                v_ax0, v_ax1, v_k2 = T.axis.remap("SSR", [ax0, ax1, k2])
                T.reads(data[v_ax0, v_ax1, v_k2])
                T.writes(data_red[v_ax0, v_ax1])
                with T.init():
                    data_red[v_ax0, v_ax1] = T.float32(0)
                data_red[v_ax0, v_ax1] = data_red[v_ax0, v_ax1] + data[v_ax0, v_ax1, v_k2]


def ms_execute(logfile, target, target_name, trials):
    # only print, just to collect the IR module
    print_reduce(shape, axis, keep_dim)

    start = time.time()
    database = ms.tune_tir(
        mod=Main,
        target=target,
        max_trials_global=trials,
        num_trials_per_iter=64,
        work_dir=logfile,
        runner=ms.runner.LocalRunner(
            evaluator_config=EvaluatorConfig(
                number=1,
                repeat=1,
                min_repeat_ms=100,
                enable_cpu_cache_flush=True if target_name == "llvm" else False,
            )
        ),
        cost_model=ms.cost_model.XGBModel(
            extractor=ms.feature_extractor.PerStoreFeature(),
            adaptive_training=False,
        ),
        strategy=ms.search_strategy.EvolutionarySearch(),
    )
    end = time.time()

    best_time = get_ms_time(logfile + "/database_tuning_record.json")

    print(f"Best time (ms): {np.mean(best_time)*1000:.10f}")
    print(f"Best std  (ms): {np.std(best_time)*1000:.10f}")
    print(f"Tuning Time (min): {(end-start)/60:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("python mm.py -a x86 -l 'results/ms/cpu_matmul'")
    parser.add_argument(
        "-a", "--arch", type=str, required=True, help="Options: x86, aarch64, cuda"
    )
    parser.add_argument("-l", "--logfile", type=str, required=True)
    parser.add_argument("-t", "--trials", type=int, default=1000)
    args = parser.parse_args()

    arch = args.arch
    logfile = args.logfile
    trials = args.trials

    # clean the files
    if os.path.isfile(logfile):
        os.remove(logfile)

    if arch == "x86":
        target_name = "llvm"
        target = tvm.target.Target(f"llvm -num-cores {num_threads // 2}")
        dev = tvm.cpu()
    elif arch == "cuda":
        target_name = "cuda"
        target = tvm.target.Target(
            "cuda -max_threads_per_block 1024 -max_shared_memory_per_block 49152"
        )
        dev = tvm.cuda()
    elif arch == "arm":
        target = tvm.target.Target("llvm -mcpu=a64fx -num-cores 48")
        dev = tvm.cpu()
    else:
        print("Archtecture doesn't support.")
        exit(0)

    ms_execute(logfile, target, target_name, trials)
