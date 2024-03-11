import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import torch
import time
import sys

S_list = [[128, 512, 1024], [65536, 1024], [128, 4032, 11, 11], [128, 2048, 7, 7]]
A_list = [[2], [1], [2,3], [2,3]]
K_list = [True, True, False, True]

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if len(sys.argv) == 3:
        id = int(sys.argv[1])
        N = S_list[id][0] 
        C = S_list[id][1]
        if id != 1:
            H = S_list[id][2]
            if id > 1:
                W = S_list[id][3]
        K = K_list[id]
        repeat_time = int(sys.argv[2])
    print("id, repeat_time:", id, repeat_time)
    
    if id == 0:
        A1 = A_list[id][0]
        a = torch.ones(size=[N, C, H], dtype=torch.float32)
        t = torch.sum(a).numpy()
        st = time.time()
        for i in range(repeat_time):
            c = torch.sum(input=a, dim=[A1], keepdim=K)
        x = torch.sum(c)
        _ = x.numpy()
        ed = time.time()
        print("{} ms on avg".format((ed-st)*1000.0/repeat_time))
    elif id == 1:
        A1 = A_list[id][0]
        a = torch.ones([N, C], torch.float32)
        t = torch.sum(a).numpy()
        st = time.time()
        for i in range(repeat_time):
            c = torch.sum(input=a, dim=[A1], keepdim=K)
        x = torch.sum(c)
        _ = x.numpy()
        ed = time.time()
        print("{} ms on avg".format((ed-st)*1000.0/repeat_time))
    else:
        A1 = A_list[id][0]
        A2 = A_list[id][1]
        a = torch.ones([N, C, H, W], torch.float32)
        t = torch.sum(a).numpy()
        st = time.time()
        for i in range(repeat_time):
            c = torch.sum(input=a, dim=[A1, A2], keepdim=K)
        x = torch.sum(c)
        _ = x.numpy()
        ed = time.time()
        print("{} ms on avg".format((ed-st)*1000.0/repeat_time))
    pass
