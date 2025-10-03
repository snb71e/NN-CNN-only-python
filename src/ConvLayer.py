import numpy as np
from src.matrix import im2col, col2im

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

        cols, OH, OW = im2col(x, FH, FW, S, P)
        W_col = self.W.reshape(self.FN, -1)
        out = W_col @ cols + self.b[:, None] 
        out = out.reshape(self.FN, N, OH, OW).transpose(1, 0, 2, 3)
        self.cache = (x.shape, cols, OH, OW, W_col)

        return out
    
    def backward(self, upstream_grad):
        x_shape, cols, OH, OW, W_col = self.cache
        N, C, H, W = x_shape
        FN, C, FH, FW = self.W.shape
        S, P = self.S, self.P

        dout2 = upstream_grad.transpose(1, 0, 2, 3).reshape(FN, N * OH * OW)

        self.db = dout2.sum(axis=1)
        self.dW = (dout2 @ cols.T).reshape(self.W.shape)

        dcols = W_col.T @ dout2
        dx = col2im(dcols, x_shape, FH, FW, stride=S, pad=P)
        return dx