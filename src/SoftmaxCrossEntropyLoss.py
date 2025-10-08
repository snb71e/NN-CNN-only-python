import numpy as np

class SoftmaxCrossEntropyLoss():
    def __init__(self):
        self.y = None
        self.t = None
        self.loss = None
    
    def softmax(self, x):
        max_x = np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x - max_x)
        sum_exp = np.sum(exp_x, axis=1, keepdims=True)
        return exp_x / sum_exp

    def cross_entropy_loss(self, y, t):
        N = y.shape[0]
        loss = -np.sum(t * np.log(y + 1e-15)) / N
        return loss
    
    def forward(self, x, t):
        self.y = self.softmax(x)
        self.t = t
        self.loss = self.cross_entropy_loss(self.y, self.t)
        return self.loss
    
    def backward(self):
        N = self.y.shape[0]
        grad_input = (self.y - self.t) / N
        return grad_input