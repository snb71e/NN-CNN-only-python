import numpy as np
from src.matrix import im2col, col2im

class MaxPooling:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

        self.cache = None
    
    def forward(self, x):
        N, C, H, W = x.shape
        kH = kW = self.pool_size
        s = self.stride

        cols, H_out, W_out = im2col(x, kH, kW, stride=s, pad=0)
        cols = cols.reshape(C, kH*kW, N*H_out*W_out) 

        max_idx = np.argmax(cols, axis=1) 
        out_vals = np.max(cols, axis=1) 

        out = out_vals.reshape(C, N, H_out, W_out).transpose(1, 0, 2, 3)

        self.cache = (x.shape, H_out, W_out, max_idx)
        return out
    
    def backward(self, upstream_grad):
        x_shape, H_out, W_out, max_idx = self.cache
        N, C, H, W = x_shape
        kH = kW = self.pool_size
        s = self.stride

        dcols = np.zeros((C, kH*kW, N*H_out*W_out), dtype=upstream_grad.dtype)
        dout_flat = upstream_grad.transpose(1, 0, 2, 3).reshape(C, N*H_out*W_out)
        for c in range(C):
            dcols[c, max_idx[c], np.arange(N*H_out*W_out)] = dout_flat[c]
        dcols = dcols.reshape(C*kH*kW, N*H_out*W_out)

        dx = col2im(dcols, x_shape, kH, kW, stride=s, pad=0)
        return dx