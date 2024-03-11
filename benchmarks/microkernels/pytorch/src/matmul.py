import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import time
import sys
import torch

if __name__ == "__main__":
    if len(sys.argv) == 5:
        M = int(sys.argv[1])
        K = int(sys.argv[2])
        N = int(sys.argv[3])
        repeat_time = int(sys.argv[4])
    print("M, K, N, repeat_time:", M, K, N, repeat_time)
    
    a = torch.ones(size=(M, K), dtype=torch.float32)
    b = torch.ones(size=(K, N), dtype=torch.float32)
    t = torch.sum(b).numpy()
    st = time.time()
    for i in range(repeat_time):
        c = torch.matmul(a, b)
    x = torch.sum(c)
    _ = x.numpy()
    ed = time.time()
    print("{} ms on avg".format((ed-st)*1000.0/repeat_time))
