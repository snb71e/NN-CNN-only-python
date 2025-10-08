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
        self.x = None

    def forward(self, x):
        self.x = x
        output = x @ self.W + self.b
        return output

    def backward(self, grad_output):
        x = self.x
        self.dW = x.T @ grad_output
        self.db = np.sum(grad_output, axis=0, keepdims=True)

        grad_input = grad_output @ self.W.T

        return grad_input