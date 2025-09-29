"""
ReLU(x) = max(0, x)
"""

class ReLU:
    def __init__(self):

        self.cache = None
    
    def forward(self, x):

        self.cache = (x <= 0)
        out = x.copy()
        out[self.cache] = 0

        return out
    
    def backward(self, upstream_grad):

        downstream_grad = upstream_grad.copy()
        downstream_grad[self.cache] = 0

        return downstream_grad