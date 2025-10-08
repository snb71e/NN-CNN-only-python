class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
    
    def update(self, params, grads):
        for x, dx in zip(params, grads):
            x -= self.learning_rate * dx