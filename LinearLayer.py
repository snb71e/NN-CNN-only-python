import numpy as np

class LinearLayer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        scale = np.sqrt(2.0 / (input_size + output_size))
        self.W = np.random.randn(input_size, output_size) * scale
        self.b = np.zeros((1, output_size))

        self.dW = None
        self.db = None
        self.cache = None

    def forward(self, X):
        self.cache = X
        output = X @ self.W + self.b
        return output

    def backward(self, upstream_grad):
        x = self.cache
        self.dW = x.T @ upstream_grad
        self.db = np.sum(upstream_grad, axis=0, keepdims=True)

        downstream_grad = upstream_grad @ self.W.T

        return downstream_grad