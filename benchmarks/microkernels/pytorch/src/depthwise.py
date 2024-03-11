import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import torch
import time
import sys

if __name__ == "__main__":
    N, C, H, W, K, S, D, P, repeat_time = 128, 84, 83, 83, 5, 2, 1, "SAME", 1000
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

    a = torch.ones(size=[N, C, H, W], dtype=torch.float32)
    b = torch.ones([C, 1, K, K], dtype=torch.float32)
    t = torch.sum(b).numpy()
    st = time.time()
    for i in range(repeat_time):
        ### Depthwise convolution is a type of convolution where we apply a single 
        # convolutional filter for each input channel
        # by adding 'groups' param, you perform depthwise conv
        depthwise_layer = torch.nn.Conv2d(in_channels=C, out_channels=N, kernel_size=K, stride=[S, S], groups=1)
        depthwise_layer_n_params = sum(p.numel() for p in depthwise_layer.parameters() if p.requires_grad)
        depthwise_out = depthwise_layer(input=a)

        #c = torch.depthwise(input=a, filter=b, strides=[1, S, S, 1], padding=P, data_format='NHWC')
    x = torch.sum(depthwise_out)
    _ = x.detach().numpy()
    ed = time.time()
    print("{} ms on avg".format((ed-st)*1000.0/repeat_time))
