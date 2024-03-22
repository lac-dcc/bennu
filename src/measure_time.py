import sys
import numpy as np

def read_ms_file(path_file):
    import json

    f = open(path_file, "r")

    info = {}
    for line in f.readlines():
        data = json.loads(line)
        
        layer = data[0]
        params = data[1]
        template = params[0]
        time = params[1]
        tensors = params[2]
        config = template[0]
        constraints = template[1]
        
        if layer not in info.keys():
            info[layer] = time
        elif info[layer] > time:
            info[layer] = time

    return info


if __name__ == "__main__":

    if len(sys.argv) > 1:
        path_file = sys.argv[1]
    else:
        print("not valid")
        exit(0)
    
    info = read_ms_file(path_file)

    for key in info:
        if np.mean(info[key]) < 100000:
            print(f"{key}, {np.mean(info[key]):.8f}, {np.std(info[key]):.8f}")
        else:
            print(f"{key}, {1000}, {0}")
    
