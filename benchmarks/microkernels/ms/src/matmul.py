import os, sys, time, argparse, tvm
from tvm import te
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
N, L, M = 1000, 800, 700
dtype = "float32"


def te_matmul(A: te.Tensor, B: te.Tensor) -> te.Tensor:
    assert A.shape[1] == B.shape[0]
    n = A.shape[0]
    m = B.shape[1]
    k = te.reduce_axis((0, A.shape[1]), name="k")
    return te.compute(
        (n, m), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="matmul"
    )


## ----------------- Benchmark -------------------
def mm_print(N, L, M, dtype="float32"):
    A = te.placeholder((N, L), name="A", dtype=dtype)
    B = te.placeholder((L, M), name="B", dtype=dtype)
    C = te_matmul(A, B)
    te.create_prim_func([A, B, C]).show()


@tvm.script.ir_module
class Main:
    @T.prim_func
    def main(
        A: T.Buffer((1000, 800), "float32"),
        B: T.Buffer((800, 700), "float32"),
        C: T.Buffer((1000, 700), "float32"),
    ):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i, j, k in T.grid(1000, 700, 800):
            with T.block("C"):
                v_i, v_j, v_k = T.axis.remap("SSR", [i, j, k])
                T.reads(A[v_i, v_k], B[v_k, v_j])
                T.writes(C[v_i, v_j])
                with T.init():
                    C[v_i, v_j] = T.float32(0)
                C[v_i, v_j] = C[v_i, v_j] + A[v_i, v_k] * B[v_k, v_j]


## ---------------------------------------------


def ms_execute(logfile, target, target_name, trials):
    # only print
    # mm_print(N, L, M, dtype)

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
