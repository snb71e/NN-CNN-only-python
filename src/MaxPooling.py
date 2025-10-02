import numpy as np

class MaxPooling:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

        self.cache = None
    
    def forward(self, x):
        N, C, H, W = x.shape
        pool_h, pool_w, stride = self.pool_size, self.pool_size, self.stride

        H_out = (H - pool_h) // stride + 1
        W_out = (W - pool_w) // stride + 1

        out = np.zeros((N, C, H_out, W_out))
        self.max_index = np.zeros_like(x, dtype=bool)

        for n in range(N):
            for c in range(C):
                for h_out in range(H_out):
                    for w_out in range(W_out):
                        h_start = h_out * stride
                        h_end = h_start + pool_h
                        w_start = w_out * stride
                        w_end = w_start + pool_w

                        window = x[n, c, h_start:h_end, w_start:w_end]
                        max_val = np.max(window)
                        out[n, c, h_out, w_out] = max_val

                        max_mask = (window == max_val)
                        self.max_index[n, c, h_start:h_end, w_start:w_end] += max_mask
        self.cache = x
        return out
    
    def backward(self, upstream_grad):
        x = self.cache
        N, C, H, W = x.shape
        dx = np.zeros((N, C, H, W))

        pool_h, pool_w, stride = self.pool_size, self.pool_size, self.stride
        H_out = upstream_grad.shape[2]
        W_out = upstream_grad.shape[3]

        for n in range(N):
            for c in range(C):
                for h_out in range(H_out):
                    for w_out in range(W_out):
                        h_start = h_out * stride
                        h_end = h_start + pool_h
                        w_start = w_out * stride
                        w_end = w_start + pool_w

                        mask = self.max_index[n, c, h_start:h_end, w_start:w_end]
                        dx[n, c, h_start:h_end, w_start:w_end] += upstream_grad[n, c, h_out, w_out] * mask                        
        return dx