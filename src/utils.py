import numpy as np
from model.ThreeLayerNN import ThreeLayerNN
from model.ThreeLayerCNN import ThreeLayerCNN

def load_model(load_path='checkpoints/nn.npz'):
    loaded = np.load(load_path)
    if all(k in loaded for k in ('input_size','hidden_size1','hidden_size2','output_size')):
        input_size  = int(loaded['input_size'])
        hidden_size1 = int(loaded['hidden_size1'])
        hidden_size2 = int(loaded['hidden_size2'])
        output_size = int(loaded['output_size'])
    else:
        input_size, hidden_size1, hidden_size2, output_size = 28*28, 128, 64, 10
    model = ThreeLayerNN(input_size, hidden_size1, hidden_size2, output_size)
    for i, param in enumerate(model.params):
        param[:] = loaded[f'param_{i}']
    return model

def load_model_cnn(load_path='checkpoints/cnn.npz'):
    loaded = np.load(load_path)
    if all(k in loaded for k in ('W1','b1','W2','b2','W3','b3','input_dim','output_size')):
        input_dim  = tuple(loaded['input_dim'])
        output_size = int(loaded['output_size'])
    else:
        input_dim, output_size = (1, 28, 28), 10
    model = ThreeLayerCNN(input_dim = input_dim, output_size=output_size)
    model.conv1.W[:] = loaded['W1']
    model.conv1.b[:] = loaded['b1']
    model.conv2.W[:] = loaded['W2']
    model.conv2.b[:] = loaded['b2']
    model.linear.W[:] = loaded['W3']
    model.linear.b[:] = loaded['b3']
    return model

def confusion_matrix(y_true, y_pred, num_classes=10):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def confusion_matrix_prob(cm_counts):
    row_sums = cm_counts.sum(axis=1, keepdims=True).astype(np.float64)
    row_sums[row_sums == 0] = 1.0
    cm_prob = cm_counts / row_sums
    return cm_prob
