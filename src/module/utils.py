import numpy as np
from itertools import permutations, product

def permutation(arr, limit):
    result = []
    for i, perm in enumerate(permutations(arr, len(arr))):
        if i >= limit:
            break
        result.append(list(perm))
    return result

def add(list, elements):
    for e in elements:
        if e not in list:
            list.append(e)

def generate_space(values, r):
    space = []
    for idx in product(range(len(values)), repeat=r):
        space.append([values[i] for i in idx])
    return space
        

def update(list, list_remove, insert_element, pos):
    # insert the new tensor in the new position
    list.insert(pos, insert_element)
    # Removed the old tensors
    for l in list_remove:
        list.remove(l)
    

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