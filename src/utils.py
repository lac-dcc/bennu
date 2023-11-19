import numpy as np
from itertools import permutations, product
import tvm, json, os, time
import tvm.contrib.graph_executor as runtime


def write_file(json_file, log="/tmp/file.json") -> str:
    with open(log, "w", encoding="utf-8") as outfile:
        outfile.write(json.dumps(json_file))
        outfile.write("\n")
    return log


def create_file(json_list : list, log="/tmp/file.json") -> str:
    with open(log, "w", encoding="utf-8") as outfile:
        for j in json_list:
            outfile.write(json.dumps(j))
            outfile.write("\n")
    return log

def clean_file(filename):
    if os.path.isfile(filename):
        os.remove(filename)

def read_file(filename):
    f = open(filename, "r")
    for l in f.readlines():
        print(l.strip())
    f.close()

def get_tasks(filename : str) -> int:
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
                best_cfg = data["i"][1][1]
        else:
            r = data["result"][0]
            if np.mean(best_avg) > np.mean(r):
                best_avg = r
                best_cfg = data["config"]["entity"]
    f.close()

    return best_avg, best_cfg


def get_best_multilayers(log):
    import json

    hash_map = dict()
    f = open(log, "r")
    for line in f.readlines():
        data = json.loads(line)
        if "i" in data:
            r = data["r"][0]
            hash = data["i"][0][0]
            cfg = data["i"][1][1]
            if hash not in hash_map or np.mean(hash_map[hash][0]) > np.mean(r):
                hash_map[hash] = (r, cfg, data)
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