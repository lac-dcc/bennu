import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import torch
import time
import sys

if __name__ == "__main__":
    N, C, H, W, K, S, D, P, repeat_time = 128, 84, 83, 83, 5, 2, 1, "SAME", 1000

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {dev}")

    if len(sys.argv) == 10:
        N = int(sys.argv[1])
        C = int(sys.argv[2])
        H = int(sys.argv[3])
        W = int(sys.argv[4])
        K = int(sys.argv[5])
        S = int(sys.argv[6])
        D = int(sys.argv[7])
        P = str(sys.argv[8])
        repeat_time = int(sys.argv[9])
    print("N, C, H, W, K, S, D, P, repeat_time:", N, C, H, W, K, S, D, P, repeat_time)

    a = torch.ones(size=[N, C, H, W], dtype=torch.float32, device=dev)
    b = torch.ones(size=[C, C, K, K], dtype=torch.float32, device=dev)
    t = torch.sum(b)
    st = time.time()
    for i in range(repeat_time):
        ### Depthwise convolution is a type of convolution where we apply a single 
        # convolutional filter for each input channel
        # by adding 'groups' param, you perform depthwise conv
        depthwise_out = torch.conv2d(input=a, weight=b, stride=S, dilation=D, groups=1).to(dev)
    x = torch.sum(depthwise_out)
    x = x.cpu()
    _ = x.detach().numpy()
    ed = time.time()
    print("{} ms on avg".format((ed-st)*1000.0/repeat_time))
