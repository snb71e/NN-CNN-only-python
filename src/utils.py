import numpy as np
from model.ThreeLayerNN import ThreeLayerNN
from model.ThreeLayerCNN import ThreeLayerCNN

def evaluate_loss_and_acc(model, dataload):
    total = 0
    total_loss = 0.0
    correct = 0
    for images, labels in dataload:
        N = images.shape[0]
        x = images.reshape(N, -1)
        t = labels
        
        loss = model.forward(x, t)
        total_loss += float(loss) * N
        
        probs = model.forward(x, t=None)
        preds = np.argmax(probs, axis=1)
        
        t_ids = np.argmax(t, axis=1) if t.ndim == 2 else t
        correct += int((preds == t_ids).sum())
        total += N
    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc


def evaluate_loss_and_acc_cnn(model, dataload):
    total = 0
    total_loss = 0.0
    correct = 0
    
    for images, labels in dataload:
        N = images.shape[0]
        x = images 
        t = labels
        
        loss = model.forward(x, t)
        total_loss += float(loss) * N
        
        probs = model.forward(x, t=None)
        preds = np.argmax(probs, axis=1)
        
        t_ids = np.argmax(t, axis=1) if t.ndim == 2 else t
        
        correct += int((preds == t_ids).sum())
        total += N
        
    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc

def save_model(model, save_path='checkpoints/nn.npz'):
    params = {}

    for i, param in enumerate(model.params):
        params[f'param_{i}'] = param

    input_size  = model.layers['L1'].W.shape[0]
    hidden_size1 = model.layers['L1'].W.shape[1]
    hidden_size2 = model.layers['L2'].W.shape[1]
    output_size = model.layers['L3'].W.shape[1]
    params['input_size'] = np.array(input_size)
    params['hidden_size1'] = np.array(hidden_size1)
    params['hidden_size2'] = np.array(hidden_size2)
    params['output_size'] = np.array(output_size)

    np.savez(save_path, **params)
    print(f"Model parameters saved to {save_path}")


def save_model_cnn(model, save_path='checkpoints/cnn.npz'):
    params = {}
    
    params['W1'] = model.conv1.W
    params['b1'] = model.conv1.b
    params['W2'] = model.conv2.W
    params['b2'] = model.conv2.b
    
    params['W3'] = model.linear.W
    params['b3'] = model.linear.b
    
    params['input_dim'] = np.array(model.conv1.W.shape[1:])
    params['output_size'] = np.array(model.linear.W.shape[1])
    

    np.savez(save_path, **params)
    print(f"Model parameters saved to {save_path}")



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
