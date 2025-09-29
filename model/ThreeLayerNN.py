from src.LinearLayer import LinearLayer
from src.ReLU import ReLU
from src.SoftmaxCrossEntropyLoss import SoftmaxCrossEntropyLoss



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