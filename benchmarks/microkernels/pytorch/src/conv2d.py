import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import torch
import time
import sys

if __name__ == "__main__":
    repeat_time = 1000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if len(sys.argv) == 11:
        N = int(sys.argv[1])
        C = int(sys.argv[2])
        H = int(sys.argv[3])
        W = int(sys.argv[4])
        F = int(sys.argv[5])
        K = int(sys.argv[6])
        K = int(sys.argv[7])
        if '(' in sys.argv[8]:
            S = tuple(list(map(int,str(sys.argv[7])[1:-1].split(';'))))
        else:
            S = int(sys.argv[8])
        D = int(sys.argv[9])
        P = str(sys.argv[10])
    print("N, C, H, W, F, K, S, D, P, repeat_time:", N, C, H, W, F, K, S, D, P, repeat_time)
    
    a = torch.ones(size=[N, C, H, W], dtype=torch.float32)
    b = torch.ones(size=[C, F, K, K], dtype=torch.float32)
    
    t = torch.sum(b).numpy()
    st = time.time()
    for i in range(repeat_time):
        c = torch.conv2d(input=a, weight=b, stride=(S,S), padding=P, dilation=(D,D))
    x = torch.sum(c)
    _ = x.numpy()
    ed = time.time()
    print("{} ms on avg".format((ed-st)*1000.0/repeat_time))
