import os, sys, time, argparse, tvm
from tvm import te, topi
from tvm.topi.nn.utils import get_pad_tuple
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
input_shape = (128, 168, 83, 83)
dtype = "float32"

# avg      128 168 83 83 1  2       VALID
# pooltype N,  CI, H, W, K, strides padding


## ----------------- Benchmark -------------------
def print_pooling(input_shape, dtype="float32"):
    A = te.placeholder(shape=input_shape, name="A", dtype=dtype)
    B = topi.nn.pool2d(
        A, (1, 1), (2, 2), (1, 1), get_pad_tuple("VALID", (1, 1)), pool_type="avg"
    )
    te.create_prim_func([A, B]).show()


@tvm.script.ir_module
class Main:
    @T.prim_func
    def main(
        A: T.Buffer((128, 168, 83, 83), "float32"),
        pool_avg: T.Buffer((128, 168, 42, 42), "float32"),
    ):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        pool_sum = T.alloc_buffer((128, 168, 42, 42))
        for ax0, ax1, ax2, ax3, rv0, rv1 in T.grid(128, 168, 42, 42, 1, 1):
            with T.block("pool_sum"):
                v_ax0, v_ax1, v_ax2, v_ax3, v_rv0, v_rv1 = T.axis.remap(
                    "SSSSRR", [ax0, ax1, ax2, ax3, rv0, rv1]
                )
                T.reads(A[v_ax0, v_ax1, v_ax2 * 2 + v_rv0, v_ax3 * 2 + v_rv1])
                T.writes(pool_sum[v_ax0, v_ax1, v_ax2, v_ax3])
                with T.init():
                    pool_sum[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(0)
                pool_sum[v_ax0, v_ax1, v_ax2, v_ax3] = (
                    pool_sum[v_ax0, v_ax1, v_ax2, v_ax3]
                    + A[v_ax0, v_ax1, v_ax2 * 2 + v_rv0, v_ax3 * 2 + v_rv1]
                )
        for ax0, ax1, ax2, ax3 in T.grid(128, 168, 42, 42):
            with T.block("pool_avg"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(pool_sum[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(pool_avg[v_ax0, v_ax1, v_ax2, v_ax3])
                T.block_attr({"schedule_rule": "meta_schedule.pool_avg"})
                pool_avg[v_ax0, v_ax1, v_ax2, v_ax3] = pool_sum[
                    v_ax0, v_ax1, v_ax2, v_ax3
                ] / T.Cast(
                    "float32",
                    (T.min(0, 41 - v_ax2) * 2 + 1) * (T.min(0, 41 - v_ax3) * 2 + 1),
                )


def ms_execute(logfile, target, target_name, trials):
    # only print, just to collect the IR module
    # print_pooling(input_shape)

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
