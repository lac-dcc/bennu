import numpy as np

def get_best_time(log):
    import json

    f = open(log, "r")
    best_avg = 9999.0
    best_cfg = {}
    for line in f.readlines():
        data = json.loads(line)
        if "r" in data:
            r = data["r"][0]
            if (np.mean(best_avg) > np.mean(r)):
                best_avg = r
                best_cfg = data["i"][1]
        else:
            r = data["result"][0]
            if (np.mean(best_avg) > np.mean(r)):
                best_avg = r
                best_cfg = data["config"]["entity"]
    f.close()

    return best_avg, best_cfg