import numpy as np
from src.matrix import im2col, col2im

class ConvLayer:
    def __init__(self, input_channels, output_channels, filter_h, filter_w, stride=1, padding=0):

        self.C_in = input_channels
        self.C_out = output_channels
        self.FH = filter_h
        self.FW = filter_w
        self.stride = stride
        self.padding = padding

        scale = np.sqrt(2.0 / (input_channels * filter_h * filter_w))
        self.W = scale * np.random.randn(output_channels, input_channels, filter_h, filter_w)
        self.b = np.zeros(output_channels)

        self.cache = None
        self.dW = None
        self.db = None

    def pad(self, x):
        if self.padding == 0:
            return x
        return np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant', constant_values=0)
    
    def forward(self, x):
        N, C, H, W = x.shape

        cols, H_out, W_out = im2col(x, self.FH, self.FW, self.stride, self.padding)
        W_col = self.W.reshape(self.C_out, -1)
        out = W_col @ cols + self.b[:, None] 
        out = out.reshape(self.C_out, N, H_out, W_out).transpose(1, 0, 2, 3)
        self.cache = (x.shape, cols, H_out, W_out, W_col)

        return out
    
    def backward(self, grad_output):
        x_shape, cols, H_out, W_out, W_col = self.cache
        N, C, H, W = x_shape
        C_out, C_in, FH, FW = self.W.shape

        dout_col = grad_output.transpose(1, 0, 2, 3).reshape(C_out, N * H_out * W_out)

        self.db = dout_col.sum(axis=1)
        self.dW = (dout_col @ cols.T).reshape(self.W.shape)

        dcols = W_col.T @ dout_col
        grad_input = col2im(dcols, x_shape, FH, FW, stride=self.stride, pad=self.padding)
        return grad_input