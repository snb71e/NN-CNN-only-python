import numpy as np

class ConvLayer:
    def __init__(self, input_channels, output_channels, filter_h, filter_w, stride=1, padding=0):

        self.FN = output_channels
        self.C = input_channels
        self.FH = filter_h
        self.FW = filter_w
        self.S = stride
        self.P = padding

        scale = np.sqrt(2.0 / (input_channels * filter_h * filter_w))
        self.W = scale * np.random.randn(output_channels, input_channels, filter_h, filter_w)
        self.b = np.zeros(output_channels)

        self.cache = None
        self.dW = None
        self.db = None

    def pad(self, x):
        if self.P == 0:
            return x
        return np.pad(x, ((0, 0), (0, 0), (self.P, self.P), (self.P, self.P)), 'constant', constant_values=0)
    
    def forward(self, x):
        N, C, H, W = x.shape
        FH, FW, S, P = self.FH, self.FW, self.S, self.P

        x_padded = self.pad(x)
        H_pad, W_pad = x_padded.shape[2:]

        OH = (H_pad - FH) // S + 1
        OW = (W_pad - FW) // S + 1

        out = np.zeros((N, self.FN, OH, OW))

        for n in range(N):
            for fn in range(self.FN):
                for oh in range(OH):
                    for ow in range(OW):
                        h_start = oh * S
                        h_end = h_start + FH
                        w_start = ow * S
                        w_end = w_start + FW

                        window = x_padded[n, :, h_start:h_end, w_start:w_end]

                        conv_result = np.sum(window * self.W[fn]) + self.b[fn]
                        out[n, fn, oh, ow] = conv_result
        self.cache = (x, x_padded)
        return out
    
    def backward(self, upstream_grad):

        
        x, x_padded = self.cache
        N, C, H, W = x.shape
        FN, C, FH, FW = self.W.shape
        S, P = self.S, self.P

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        dx_padded = np.zeros_like(x_padded)

        N, FN, OH, OW = upstream_grad.shape

        for n in range(N):
            for fn in range(FN):
                self.db[fn] += np.sum(upstream_grad[n, fn])
                for oh in range(OH):
                    for ow in range(OW):
                        h_start = oh * S
                        h_end = h_start + FH
                        w_start = ow * S
                        w_end = w_start + FW

                        window = x_padded[n, :, h_start:h_end, w_start:w_end]

                        grad_value = upstream_grad[n, fn, oh, ow]

                        self.dW[fn] += window * grad_value
                        dx_padded[n, :, h_start:h_end, w_start:w_end] += self.W[fn] * grad_value
        if P == 0:
            downstream_grad = dx_padded
        else:
            downstream_grad = dx_padded[:, :, P:H+P, P:W+P]
        return downstream_grad