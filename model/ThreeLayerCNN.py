from src.ConvLayer import ConvLayer
from src.ReLU import ReLU
from src.MaxPooling import MaxPooling
from src.LinearLayer import LinearLayer
from src.SoftmaxCrossEntropyLoss import SoftmaxCrossEntropyLoss 


class ThreeLayerCNN:
    def __init__(self, input_dim=(1, 28, 28), 
                 conv1_params={'filter_num': 16, 'filter_size': 3, 'padding': 1, 'stride': 1},
                 pool1_params={'pool_size': 2, 'stride': 2},
                 conv2_params={'filter_num': 32, 'filter_size': 3, 'padding': 1, 'stride': 1},
                 pool2_params={'pool_size': 2, 'stride': 2},
                 output_size=10):
        
        C_in, H_in, W_in = input_dim

        self.conv1 = ConvLayer(
            C_in, 
            conv1_params['filter_num'], 
            conv1_params['filter_size'], 
            conv1_params['filter_size'], 
            conv1_params['stride'], 
            conv1_params['padding']
        )
        self.relu1 = ReLU()
        
        H_c1 = 28
        W_c1 = 28
        self.pool1 = MaxPooling(pool1_params['pool_size'], pool1_params['stride'])
        H_p1 = H_c1 // pool1_params['stride'] # 28 / 2 = 14
        W_p1 = W_c1 // pool1_params['stride'] # 28 / 2 = 14

        self.conv2 = ConvLayer(
            conv1_params['filter_num'], 
            conv2_params['filter_num'], 
            conv2_params['filter_size'], 
            conv2_params['filter_size'], 
            conv2_params['stride'], 
            conv2_params['padding']
        )
        self.relu2 = ReLU()
        
        H_c2 = 14 
        W_c2 = 14
        self.pool2 = MaxPooling(pool2_params['pool_size'], pool2_params['stride'])
        H_p2 = H_c2 // pool2_params['stride'] # 14 / 2 = 7
        W_p2 = W_c2 // pool2_params['stride'] # 14 / 2 = 7

        self.H_p2 = H_p2
        self.W_p2 = W_p2

        self.flatten_size = conv2_params['filter_num'] * H_p2 * W_p2 # 32 * 7 * 7 = 1568
        self.linear = LinearLayer(self.flatten_size, output_size)

        self.layers = [
            self.conv1, self.relu1, self.pool1,
            self.conv2, self.relu2, self.pool2,
            self.linear
        ]
        self.last_layer = SoftmaxCrossEntropyLoss()
        
        self.params = []
        for layer in [self.conv1, self.conv2, self.linear]:
            self.params.append(layer.W)
            self.params.append(layer.b)
        
        self.grads = [None] * len(self.params)


    def forward(self, x, t=None):

        for i, layer in enumerate(self.layers[:-1]):
            x = layer.forward(x)
            
            if isinstance(layer, MaxPooling):
                if layer == self.pool2:
                    N, C, H, W = x.shape
                    x = x.reshape(N, -1)
        
        output = self.linear.forward(x)

        if t is not None:
            loss = self.last_layer.forward(output, t)
            return loss
        
        return self.last_layer.softmax(output)
    
    def backward(self):

        dout = self.last_layer.backward()
        dout = self.linear.backward(dout) 
        
        N, _ = dout.shape
        dout = dout.reshape(N, self.conv2.C_out, self.H_p2, self.W_p2)

        reversed_layers = self.layers[:-1]
        reversed_layers.reverse()
        
        for layer in reversed_layers:
            dout = layer.backward(dout)
        
        self.grads = []
        for layer in [self.conv1, self.conv2, self.linear]:
            self.grads.append(layer.dW)
            self.grads.append(layer.db)