import time
from tvm import auto_scheduler
from tvm.auto_scheduler.search_task import SearchTask

def execute_ansor(workload_key, log_file, target, trials):
    task = SearchTask(workload_key=workload_key, target=target)

    ## Set Parameters for Auto-Scheduler
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=trials, # change this to 20000 to achieve the best performance
        runner=auto_scheduler.LocalRunner(
            number=10,
            repeat=3,
            timeout=100,
            enable_cpu_cache_flush=True if target == "llvm" else False,
        ),
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=0,
    )

    start = time.time()
    # Run auto-tuning (search)
    task.tune(tune_option)
    end = time.time()

    f = open(log_file, "r")
    for l in f.readlines():
        print(l.strip())