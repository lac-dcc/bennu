import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import torch
import time
import sys

if __name__ == "__main__":
    if len(sys.argv) == 9:
        N = int(sys.argv[1])
        C = int(sys.argv[2])
        H = int(sys.argv[3])
        W = int(sys.argv[4])
        K = int(sys.argv[5])
        S = int(sys.argv[6])
        P = str(sys.argv[7])
        repeat_time = int(sys.argv[8])
    print("N, C, H, W, K, S, P, repeat_time:", N, C, H, W, K, S, P, repeat_time)
    
    a = torch.ones(size=[N, H, W, C], dtype=torch.float32)
    t = torch.sum(a).numpy()
    st = time.time()
    for i in range(repeat_time):
        pool = torch.nn.AvgPool2d(kernel_size=K, stride=(S, S))
        c = pool(a)
    x = torch.sum(c)
    _ = x.numpy()
    ed = time.time()
    print("{} ms on avg".format((ed-st)*1000.0/repeat_time))
