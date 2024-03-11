import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import torch
import time
import sys

C = 4096
repeat_time = 1000

if __name__ == "__main__":

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {dev}")

    if len(sys.argv) == 2:
        N = int(sys.argv[1])
    print("N, C, repeat_time:", N, C, repeat_time)

    a = torch.ones(size=[N, C], dtype=torch.float32, device=dev)
    t = torch.sum(a)
    st = time.time()
    for i in range(repeat_time):
        c = torch.relu(a)
    x = torch.sum(c)
    x = x.cpu()
    _ = x.numpy()
    ed = time.time()
    print("{} ms on avg".format((ed-st)*1000.0/repeat_time))
    pass
