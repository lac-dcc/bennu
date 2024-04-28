import tvm, argparse, os, sys, time, onnx
from scipy import stats
from tvm.driver import tvmc
from tvm.driver.tvmc.autotuner import autoscheduler_get_tuning_tasks

# from tvm.auto_scheduler.task_scheduler import droplet_exploitation

# meta schedule
from tvm import meta_schedule as ms
from tvm.target import Target
from tvm.relay.frontend import from_onnx
from tvm.meta_schedule.runner.config import EvaluatorConfig

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.utils import *
from src.DropletSearch import Droplet
from src.GridSearch import GridSearch
from src.DPMeta import DropletMeta

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


def generate_meta_template(bench, logfile, target_name, trials):
    target = tvm.target.Target(target_name)
    if target_name == "cuda":
        target = "cuda -max_threads_per_block 1024 -max_shared_memory_per_block 49152"
    mod, params = from_onnx(onnx.load(bench))

    start = time.time()
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
            cost_model=ms.cost_model.XGBModel(
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
    end = time.time()
    print(f"Tuning Time (min): {(end-start)/60:.2f}")
    print(profiler.table())


def build_meta_template(bench, logfile, target_name, top, trials):
    mod, params = from_onnx(onnx.load(bench))
    target = tvm.target.Target(target_name)

    ms_tuning_file = logfile + "/database_tuning_record.json"
    ms_workload_file = logfile + "/database_workload.json"
    output_file = logfile + "/output.txt"

    # calculate how many time spent each sample in MS
    each_sample_time = get_time_spent(output_file, ms_tuning_file)

    cfg_10k = read_ms_file(ms_tuning_file, ms_workload_file)
    cfg_top = read_ms_file(ms_tuning_file, ms_workload_file, top)

    print(
        f"Layer, DPMeta time(s), DPMeta std(s), DPMeta trials, DPMeta Tuning(min), DPMeta+Meta tuning(min), Meta-{top} time(s), Meta-{top} std(s), trials-{top}, Meta 10k time(s), Meta 10k std(s), trials-10k, speedup-{top}, speedup-10k"
    )
    for layer in cfg_top:
        ms_time, ms_cfg, ms_workload, ms_trials = cfg_top[layer]
        ms_10k_time, _, _, ms_10k_trials = cfg_10k[layer]

        # if layer != 11:
        #    continue

        log = f"layer_{layer}.log"
        clean_file(log)

        m = DropletMeta(ms_cfg, ms_workload, target, log)
        start = time.time()
        m.tune(trials)
        end = time.time() - start

        ms_time_tuning = end / 60.0

        dp_time, dp_trials = get_data_ms(log)

        mean_ms_time = np.mean(ms_time)
        std_ms_time = np.std(ms_time)
        mean_ms_10k_time = np.mean(ms_10k_time)
        std_ms_10k_time = np.std(ms_10k_time)
        mean_time = np.mean(dp_time)
        std_time = np.std(dp_time)

        total_time_ms = min(top, ms_10k_trials) * each_sample_time + ms_time_tuning

        speedup = mean_ms_time / mean_time
        speedup_10k = mean_ms_10k_time / mean_time

        print(
            f"{layer}, {mean_time:.10f}, {std_time:.10f}, {dp_trials}, {ms_time_tuning:.2f}, {total_time_ms:.2f}, {mean_ms_time:.10f}, {std_ms_time:.10f}, {ms_trials}, {mean_ms_10k_time:.10f}, {std_ms_10k_time:.10f}, {ms_10k_trials}, {speedup:.2f}, {speedup_10k:.2f}"
        )


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
        f"Layer, Time Droplet (s), std droplet (s), Tuning time Droplet (s), Tuning time Droplet+Ansor (s), tasks Droplet, Time Ansor-{top} (s), std Ansor-{top}, tuning time Ansor-{top}, task Ansor-{top}, Time Ansor 10k (s), std Ansor 10k, tuning time 10k (s), tasks Ansor, speedup top-{top}, speedup 10k, speed tuning time top-{top}, speed tuning time 10k, p-value"
    )

    for layer, workload in enumerate(cfg):
        if index != -1 and layer != index:
            continue

        log = f"layer_{layer}.log"
        clean_file(log)

        _, _, json_file = cfg[workload]
        t, _, _ = cfg_10k[workload]  # get the best value in 10k
        if method == "droplet":
            m = Droplet(json_file, target, log)
        elif method == "grid":
            m = GridSearch(json_file, target, log)
        else:
            raise (f"Method {method} is not implemeted yet")

        m.tune(n_trial=trials)

        time_m, _ = get_time_total(log)
        m_avg, m_cfg = get_best_time(log)
        top_avg, _, _ = cfg[workload]
        task_ansor = get_task_multilayers(logfile)[workload]

        time_ansor = task_ansor * time_each_point_ansor
        time_ansor_m = time_m + min(top, task_ansor) * time_each_point_ansor
        pvalue = stats.ttest_ind(np.array(m_avg), np.array(t)).pvalue

        print(
            "%d, %.8f, %.8f, %.2f, %.2f, %d, %.8f, %.8f, %.2f, %2d, %.8f, %.8f, %.2f, %d, %.2f, %.2f, %.2f, %.2f, %.8f"
            % (
                layer,
                np.mean(m_avg),
                np.std(m_avg),
                time_m,
                time_ansor_m,
                get_tasks(log),
                np.mean(top_avg),
                np.std(top_avg),
                min(top, task_ansor) * time_each_point_ansor,
                min(top, task_ansor),
                np.mean(t),
                np.std(t),
                time_ansor,
                task_ansor,
                np.mean(top_avg) / np.mean(m_avg),
                np.mean(t) / np.mean(m_avg),
                min(top, task_ansor) * time_each_point_ansor / time_ansor_m,
                time_ansor / time_ansor_m,
                pvalue,
            )
        )
        append_file(m_cfg, m_log)


def run(logfile, bench, target, dev):
    model = tvmc.load(bench)
    package = tvmc.compile(model, target=target, tuning_records=logfile)
    result = tvmc.run(package, device=dev)
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "python models_onnx.py -m ansor -a x86 -l results/model.json -b models/resnet18.onnx"
    )
    parser.add_argument(
        "-m", "--method", type=str, required=True, help="Options: ansor, droplet, grid"
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
    elif method == "dpansor":
        build_template(bench, logfile, index, target, trials, top, method)
    elif method == "dpmeta":
        build_meta_template(bench, logfile, target_name, top, trials)
    elif method == "meta":
        generate_meta_template(bench, logfile, target_name, trials)
    elif method == "run":
        run(logfile, target, dev)
