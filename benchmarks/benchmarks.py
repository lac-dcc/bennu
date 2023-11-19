import time, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.DropletSearch import Droplet
from src.utils import *

def build_template(name, logfile, target, trials):
    t_ansor, workload, json_file = get_best_template(logfile)

    print("Layer, Time Droplet (s), Tuning time Droplet (s), tasks Droplet, Time Ansor (s), tasks Ansor, speedup")

    log = name +".log"
    clean_file(log)

    droplet = Droplet(json_file, workload, target, log, trials)
    start = time.time()
    droplet.tune()
    end = time.time()

    droplet_avg, _ = get_best_time(log)
            
    print(
        "%s, %.7f, %.2f, %d, %.7f, %d, %.2f"
        % (
            name,
            np.mean(droplet_avg),
            end - start,
            get_tasks(log),
            np.mean(t_ansor),
            get_task_multilayers(logfile)[workload],
            np.mean(t_ansor) / np.mean(droplet_avg),
        )
    )