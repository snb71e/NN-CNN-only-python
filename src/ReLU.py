"""
ReLU(x) = max(0, x)
"""
import numpy as np

class ReLU:
    def __init__(self):
        self.mask = None
    
    def forward(self, x):
        self.mask = (x <= 0)
        out = np.maximum(0, x)

        return out
    
    def backward(self, grad_output):
        grad_input = grad_output.copy()
        grad_input[self.mask] = 0

        return grad_input