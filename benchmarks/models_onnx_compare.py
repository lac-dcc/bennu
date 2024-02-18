import tvm, argparse, os, sys, time, onnx
from scipy import stats
from tvm.driver import tvmc
from tvm.driver.tvmc.autotuner import autoscheduler_get_tuning_tasks

# from tvm.auto_scheduler.task_scheduler import droplet_exploitation

# meta schedule
from tvm import meta_schedule as ms
from tvm.relay.frontend import from_onnx
from tvm.meta_schedule.runner.config import EvaluatorConfig

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.utils import *
from src.DropletSearch import Droplet
from src.GridSearch import GridSearch
from src.RandomSearch import RandomSearch
from src.GASearch import GASearch
from src.XGBSearch import XGBSearch

num_threads = os.cpu_count()
os.environ["TVM_NUM_THREADS"] = str(num_threads)
os.environ["MKL_NUM_THREADS"] = str(num_threads * 2 // 3)
os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads * 2 // 3)
os.environ["OMP_NUM_THREADS"] = str(num_threads * 2 // 3)


def generate_ansor_template(bench, logfile, target, trials):
    model = tvmc.load(bench)
    clean_file(logfile)
    start = time.time()
    tvmc.tune(
        tvmc_model=model,
        target=target,
        tuning_records=logfile,
        repeat=3,
        timeout=100 if target != "cuda" else 10,
        parallel=os.cpu_count(),
        trials=trials,
        enable_autoscheduler=True,
        # verbose=0
    )
    # droplet_exploitation(logfile, target)
    end = time.time()
    print("time search:", end - start)


def generate_meta_template(bench, logfile, target, trials):
    target = target + " -num-cores 8"  # Think to get this value automatically
    mod, params = from_onnx(onnx.load(bench))

    with ms.Profiler() as profiler:
        database = ms.relay_integration.tune_relay(
            mod=mod,
            target=target,
            params=params,
            work_dir=logfile,
            max_trials_global=trials,
            num_trials_per_iter=64,
            runner=ms.runner.LocalRunner(
                evaluator_config=EvaluatorConfig(
                    number=3,
                    repeat=3,
                    min_repeat_ms=100,
                    enable_cpu_cache_flush=False,
                )
            ),
            cost_model=ms.cost_model.XGBModel(  # type: ignore
                extractor=ms.feature_extractor.PerStoreFeature(),
                adaptive_training=False,
            ),
            strategy=ms.search_strategy.EvolutionarySearch(),
        )
        lib = ms.relay_integration.compile_relay(
            database=database,
            mod=mod,
            target=target,
            params=params,
        )
    print("Tuning Time:")
    print(profiler.table())


def build_template(bench, logfile, index, target, trials, top=1000, method="droplet"):
    model = tvmc.load(bench)
    tasks, weights = autoscheduler_get_tuning_tasks(
        mod=model.mod, params=model.params, target=target
    )

    m_log = ".".join(logfile.split(".")[:-1]) + "_droplet.json"
    clean_file(m_log)

    cfg = get_best_multilayers(logfile, top)
    cfg_10k = get_best_multilayers(logfile, 10000)
    _, time_each_point_ansor = get_time_total(logfile)

    print(
        f"layer, droplet-{top} exec, droplet-{top} tuning, droplet-{top} trials, {method}-{top} exec, {method}-{top} tuning, {method}-{top} trials, exec speedup, tuning speedup"
    )

    for layer, workload in enumerate(cfg):
        if index != -1 and layer != index:
            continue

        log = f"layer_{layer}.log"
        clean_file(log)

        _, _, json_file = cfg[workload]

        if method == "droplet":
            m = Droplet(json_file, target, log)
        elif method == "grid":
            m = GridSearch(json_file, target, log)
        elif method == "random":
            m = RandomSearch(json_file, target, log)
        elif method == "ga":
            m = GASearch(json_file, target, log)
        elif method == "xgb":
            m = XGBSearch(json_file, target, log)
        else:
            raise (f"Method {method} is not implemeted yet")

        start = time.time()
        m.tune(n_trial=trials)
        m_time = max(2, time.time() - start)

        _, m_trials = get_time_total(log)
        m_avg, _ = get_best_time(log)

        droplet_log = f"droplet_layer_{layer}.log"
        clean_file(droplet_log)

        droplet = Droplet(json_file, target, droplet_log)
        start = time.time()
        droplet.tune(n_trial=trials)
        time_droplet = time.time() - start

        _, trials_droplet = get_time_total(droplet_log)
        avg_droplet, _ = get_best_time(droplet_log)

        print(
            "%d, %.8f, %.2f, %d, %.8f, %.2f, %d, %.2f, %.2f"
            % (
                layer,
                np.mean(avg_droplet),
                time_droplet,
                trials_droplet,
                np.mean(m_avg),
                m_time,
                m_trials,
                np.mean(avg_droplet) / np.mean(m_avg),
                time_droplet / m_time,
            )
        )


def run(logfile, bench, target, dev):
    model = tvmc.load(bench)
    package = tvmc.compile(model, target=target, tuning_records=logfile)
    result = tvmc.run(package, device=dev)
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "python print_record_info.py -m ansor -a x86 -l results/model.json -i 3"
    )
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        required=True,
        help="Options: ansor, droplet, grid, random",
    )
    parser.add_argument(
        "-a", "--arch", type=str, required=True, help="Options: x86, arm, cuda"
    )
    parser.add_argument("-l", "--logfile", type=str, required=True)
    parser.add_argument("-b", "--benchmark", type=str, required=True)
    parser.add_argument("-i", "--index", type=int, default=-1)
    parser.add_argument("-t", "--trials", type=int, default=100)
    parser.add_argument("-k", "--top", type=int, default=1000)
    args = parser.parse_args()

    method = args.method
    arch = args.arch
    logfile = args.logfile
    bench = args.benchmark
    index = args.index
    trials = args.trials
    top = args.top

    if arch == "x86":
        target_name = "llvm"
        target = tvm.target.Target("llvm")
        dev = tvm.cpu()
    elif arch == "cuda":
        target_name = "cuda"
        target = tvm.target.Target("cuda")
        dev = tvm.cuda()
    elif arch == "arm":
        target_name = "llvm -mcpu=a64fx"
        target = tvm.target.Target("llvm -mcpu=a64fx")
        dev = tvm.cpu()
    else:
        print("Archtecture doesn't support.")
        exit(0)

    if method == "ansor":
        generate_ansor_template(bench, logfile, target_name, trials)
    elif method in ["droplet", "grid", "random", "ga", "xgb"]:
        build_template(bench, logfile, index, target, trials, top, method)
    elif method == "meta":
        generate_meta_template(bench, logfile, target_name, trials)
    elif method == "run":
        run(logfile, target, dev)
