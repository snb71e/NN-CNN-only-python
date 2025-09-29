"""
[Pipeline]
    1. Implement stochastic gradient descent (SGD)
    2. Training pipeline
        - Initialize the model parameters
        - Implement and do forward propagation
        - Implement and compute the cross-entropy loss
        - Implement and do backward propagation
        - Implement and update model parameter using SGD
    3. Train a 3-layer NN
"""

import numpy as np
from LinearLayer import LinearLayer
from ReLU import ReLU
from SoftmaxCrossEntropyLoss import SoftmaxCrossEntropyLoss
from dataloader import Dataloader

class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
    
    def update(self, params, grads):
        for param, grad in zip(params, grads):
            param -= self.learning_rate * grad

class ThreeLayerNN:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        self.layers = {
            'L1': LinearLayer(input_size, hidden_size1),
            'R1': ReLU(),
            'L2': LinearLayer(hidden_size1, hidden_size2),
            'R2': ReLU(),
            'L3': LinearLayer(hidden_size2, output_size)
        }

        self.last_layer = SoftmaxCrossEntropyLoss()

        self.params = []
        self.grads = []

        for name in ['L1', 'L2', 'L3']:
            self.params.append(self.layers[name].W)
            self.params.append(self.layers[name].b)
            self.grads.append(self.layers[name].dW)
            self.grads.append(self.layers[name].db)
    
    def forward(self, x, t):

        output = x
        for name, layer in self.layers.items():
            output = layer.forward(output)
        
        if t is not None:
            loss = self.last_layer.forward(output, t)
            return loss
        
        return self.last_layer.softmax(output)
    
    def backward(self):
        dout = self.last_layer.backward()

        layers_names = list(self.layers.keys())
        layers_names.reverse()

        for name in layers_names:
            layer = self.layers[name]
            dout = layer.backward(dout)
        
        self.grads = []
        for name in ['L1', 'L2', 'L3']:
            self.grads.append(self.layers[name].dW)
            self.grads.append(self.layers[name].db)

def train(model, dataloader):
    batch_size = 128
    learning_rate = 0.01
    epochs = 5

    dataload = Dataloader(dataloader, is_train=True, shuffle=True, batch_size=batch_size)
    optimizer = SGD(learning_rate=learning_rate)

    for epoch in range(epochs):
        total_loss = 0

        for batch_idx, (images, labels) in enumerate(dataload):
            N = images.shape[0]
            x_batch = images.reshape(N, -1)
            t_batch = labels

            loss = model.forward(x_batch, t_batch)
            total_loss += loss * N

            model.backward()

            optimizer.update(model.params, model.grads)

            if (batch_idx + 1) % 100 == 0:
                avg_loss = total_loss / ((batch_idx + 1) * N)
                print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(dataload)}], Loss: {avg_loss:.4f}")

        avg_loss = total_loss / len(dataload.images)
        print(f"Epoch [{epoch+1}/{epochs}] completed. Average Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    input_size = 28 * 28
    hidden_size1 = 128
    hidden_size2 = 64
    output_size = 10

    model = ThreeLayerNN(input_size, hidden_size1, hidden_size2, output_size)

    train(model, dataloader='dataset')