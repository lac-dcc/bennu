import numpy as np
from itertools import permutations, product
import tvm, json, os, time
import tvm.contrib.graph_executor as runtime


def append_file(json_file, log="/tmp/file.json") -> str:
    with open(log, "a", encoding="utf-8") as outfile:
        outfile.write(json.dumps(json_file))
        outfile.write("\n")
    return log


def create_file(json_list: list, log="/tmp/file.json") -> str:
    with open(log, "w", encoding="utf-8") as outfile:
        for j in json_list:
            outfile.write(json.dumps(j))
            outfile.write("\n")
    return log


def write_file(json_list: list, log="/tmp/file.json", mode="w") -> str:
    """Write the log file

    Parameters
    ----------
    json_list: list
        The list input json
    log: Optional[str]
        Path destiny to save the log file
    mode: Optional[str]
        Mode save, "a" means append and "w" means write

    Returns
    -------
    ret: str
        log path file
    """
    print(log)
    with open(log, mode, encoding="utf-8") as outfile:
        for j in json_list:
            outfile.write(json.dumps(j) + "\n")
    return log


def get_time(log):
    """Colect the time from log file

    Parameters
    ----------
    log: str
        The input log path with the Ansor parameter

    Returns
    -------
    ret: Union[float, float, dict]
        Returns the best time, total time, and data
    """
    time_total, best_time, best_cfg = 0, 1e10, {}
    with open(log, "r", encoding="utf-8") as log_file:
        for line in log_file.readlines():
            data = json.loads(line)
            if "r" in data:
                res = data["r"][0]
                time_total += data["r"][2]
                if np.mean(res) < np.mean(best_time):
                    best_time, best_cfg = res, data
    return best_time, time_total, best_cfg


def read_ms_file(path_tuning_file, path_workload_file):
    """Colect the info from meta logfile

    Parameters
    ----------
    log: str
        The input log path with the meta parameter

    Returns
    -------
    ret: dict[layer, Union[time, dict]]
        Returns the best time, total time, and data
    """
    import json

    workload_list = []
    with open(path_workload_file, "r", encoding="utf-8") as log_file:
        for line in log_file.readlines():
            data = json.loads(line)
            workload_list.append(data)

    info = dict()
    with open(path_tuning_file, "r", encoding="utf-8") as log_file:
        for line in log_file.readlines():
            data = json.loads(line)
            layer = data[0]
            params = data[1]
            template = params[0]
            time = params[1]
            tensors = params[2]
            config = template[0]
            constraints = template[1]
            if layer not in info.keys() or np.mean(info[layer][0]) > np.mean(time):
                info[layer] = [time, data, workload_list[layer]]
    return info


def get_ms_time(log):
    best_time = [9999]
    with open(log, "r", encoding="utf-8") as log_file:
        for line in log_file.readlines():
            data = json.loads(line)
            params = data[1]
            time = params[1]
            if np.mean(best_time) > np.mean(time):
                best_time = time
    return best_time


def clean_file(filename):
    if os.path.isfile(filename):
        os.remove(filename)


def read_file(filename):
    f = open(filename, "r")
    for l in f.readlines():
        print(l.strip())
    f.close()


def get_tasks(filename: str) -> int:
    f = open(filename, "r")
    count = len(f.readlines())
    f.close()
    return count


def permutation(arr, limit):
    result = []
    for i, perm in enumerate(permutations(arr, len(arr))):
        if i >= limit:
            break
        result.append(list(perm))
    return result


def add_space(list, element, limit=10000):
    new_list = element
    for l in list:
        if l not in new_list and l <= limit:
            new_list.append(l)
    return new_list


def add(list, elements):
    for e in elements:
        list.append(e)


def add_unique(list, elements):
    for e in elements:
        if e not in list:
            list.append(e)


def insert(list, elements, pos):
    for e in elements:
        list.insert(pos, e)
        pos += 1
    # remove the first element which was splited from the list
    try:
        del list[pos]
    except:
        print(f"Warning: element in position {pos} not deleted!")
        pass


def generate_space(values, r):
    space = []
    for idx in product(range(len(values)), repeat=r):
        space.append([values[i] for i in idx])
    return space


def update(list, list_remove, insert_element, pos):
    # insert the new axis in the new position
    list.insert(pos, insert_element)
    # Removed the old tensors
    for l in list_remove:
        list.remove(l)


def print_list(list):
    for i in range(len(list)):
        print(i, list[i])


def updateAxis(axis, new_axis, elem_del):
    for ax in new_axis:
        axis.append(ax)
    del axis[elem_del]


def limited_interval(self, max_value, interval):
    new_interval = []
    for elem in interval:
        if max_value <= elem:
            continue
        new_interval.append(elem)
    return new_interval


def get_best_time(log):
    import json

    f = open(log, "r")
    best_avg = 9999.0
    best_cfg = {}
    for line in f.readlines():
        data = json.loads(line)
        if "r" in data:
            r = data["r"][0]
            if np.mean(best_avg) > np.mean(r):
                best_avg = r
                best_cfg = data
        else:
            r = data["result"][0]
            if np.mean(best_avg) > np.mean(r):
                best_avg = r
                best_cfg = data
    f.close()

    return best_avg, best_cfg


def get_first_time(log):
    import json

    f = open(log, "r")
    for line in f.readlines():
        data = json.loads(line)
        if "r" in data:
            r = data["r"][0]
        else:
            r = data["result"][0]
        f.close()
        return r, data
    f.close()


def get_time_total(log):
    import json

    time_total, count = 0, 0
    f = open(log, "r")
    for line in f.readlines():
        data = json.loads(line)
        if "r" in data:
            time_total += data["r"][2]
        count += 1
    f.close()
    return time_total, count


def get_best_multilayers(log, top=1000):
    import json

    hash_map = dict()
    count = dict()
    f = open(log, "r")
    for line in f.readlines():
        data = json.loads(line)
        if "i" in data:
            r = data["r"][0]
            hash = data["i"][0][0]
            cfg = data["i"][1][1]

            # limit the number of task
            if hash not in count:
                count[hash] = 1
            elif count[hash] > top:
                continue
            else:
                count[hash] += 1

            if hash not in hash_map or np.mean(hash_map[hash][0]) > np.mean(r):
                hash_map[hash] = (r, cfg, data)
    f.close()
    return hash_map


def get_hash(value):
    return value.split(",")[0].replace('["', "").replace('"', "")


def reuse_cache():
    pass


def get_best_multilayers_cache(log, top=1000):
    import json

    hash_map = dict()
    name_list = []
    count = dict()

    f = open(log, "r")
    for line in f.readlines():
        data = json.loads(line)
        if "i" in data:
            r = data["r"][0]
            name = data["i"][0][0]
            hash = get_hash(name)
            cfg = data["i"][1][1]

            for l in name_list:
                # verify if already have the same hash before
                if hash in l and name not in hash_map:
                    new_params = hash_map[l][2].copy()
                    new_params["i"][0][0] = name
                    hash_map[name] = ([10000], cfg, data)
                    count[name] = 100000
                    name_list.append(name)
                    break

            # limit the number of task
            if name not in count:
                count[name] = 1
                name_list.append(name)
            elif count[name] > top:
                continue
            else:
                count[name] += 1

            if name not in hash_map or np.mean(hash_map[name][0]) > np.mean(r):
                hash_map[name] = (r, cfg, data)

    f.close()
    return hash_map


def get_task_multilayers(log):
    import json

    hash_map = dict()
    f = open(log, "r")
    for line in f.readlines():
        data = json.loads(line)
        if "i" in data:
            hash = data["i"][0][0]
            if hash not in hash_map:
                hash_map[hash] = 1
            else:
                hash_map[hash] += 1
    f.close()
    return hash_map


def get_best_template(log):
    import json

    hash_map = []
    f = open(log, "r")
    for line in f.readlines():
        data = json.loads(line)
        if "i" in data:
            r = data["r"][0]
            hash = data["i"][0][0]
            cfg = data["i"][1][1]
            if len(hash_map) == 0 or np.mean(hash_map[0]) > np.mean(r):
                hash_map = [r, hash, data]
    f.close()
    return hash_map


def evaluate_performance(lib, data_shape, target, input_name="data", dtype="float32"):
    # upload parameters to device
    dev = tvm.device(str(target), 0)
    np.random.seed(0)
    data_tvm = tvm.nd.array(
        (np.random.uniform(size=data_shape)).astype(dtype), device=dev
    )
    module = runtime.GraphModule(lib["default"](dev))
    module.set_input(input_name, data_tvm)

    # Create graph runtime
    # ctx = tvm.device(str(target), 0)
    # module = runtime.GraphModule(lib["default"](ctx))
    # data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype("int64"), device=ctx)
    # module.set_input("input_ids", data_tvm)

    r = []
    for i in range(3):
        eval = module.benchmark(
            dev, number=5, repeat=5, min_repeat_ms=100, cooldown_interval_ms=100
        )
        r.append(eval.mean * 1000)
    return r


def convert_to_list(cfg):
    cfg_list = []
    for c in cfg:
        tmp = []
        for i in c:
            if type(i) == int or type(i) == str:
                tmp.append(i)
            else:
                tmp.append(list(i))
        cfg_list.append(tmp)
    return cfg_list


def build_template(name, logfile, target, trials):
    from src.DropletSearch import Droplet

    t_ansor, workload, json_file = get_best_template(logfile)

    print(
        "Layer, Time Droplet (s), Tuning time Droplet (s), tasks Droplet, Time Ansor (s), tasks Ansor, speedup"
    )

    log = name + ".log"
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
