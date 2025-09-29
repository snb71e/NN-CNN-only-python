import numpy as np

class MaxPooling:
    def __init__(self, pool_h, pool_w, stride=1, padding=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.padding = padding

        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        PH, PW = self.pool_h, self.pool_w
        S = self.stride

        out_h = (H - PH) // S + 1
        out_w = (W - PW) // S + 1

        out = np.zeros((N, C, out_h, out_w))
        self.arg_max = np.zeros((N, C, out_h, out_w), dtype=np.int)

        for n in range(N):
            for c in range(C):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * S
                        h_end = h_start + PH
                        w_start = j * S
                        w_end = w_start + PW

                        window = x[n, c, h_start:h_end, w_start:w_end]

                        out[n, c, i, j] = np.max(window)

                        self.arg_max[n, c, i, j] = np.argmax(window)
        return out
    
    def backward(self, upstream_grad):
        N, C, out_h, out_w = upstream_grad.shape
        PH, PW = self.pool_h, self.pool_w
        S = self.stride
        
        H_in = (out_h - 1) * S + PH
        W_in = (out_w - 1) * S + PW

        downstream_grad = np.zeros((N, C, H_in, W_in))
        

        for n in range(N):
            for c in range(C):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = out_h * S
                        h_end = h_start + PH
                        w_start = out_w * S
                        w_end = w_start + PW

                        arg_max_idx = self.arg_max[n, c, out_h, out_w]

                        grad_value = upstream_grad[n, c, out_h, out_w]
                        
                        mask = np.zeros(PH * PW)
                        mask[arg_max_idx] = grad_value
                        
                        mask = mask.reshape(PH, PW)
                        
                        downstream_grad[n, c, h_start:h_end, w_start:w_end] += mask

        return downstream_grad