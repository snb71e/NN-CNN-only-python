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
from src.LinearLayer import LinearLayer
from src.ReLU import ReLU
from src.SoftmaxCrossEntropyLoss import SoftmaxCrossEntropyLoss
from src.dataloader import Dataloader
from src.SGD import SGD
import os
import json

from model.ThreeLayerNN import ThreeLayerNN



def evaluate_loss_and_acc(model, dataload):
    total = 0
    total_loss = 0.0
    correct = 0
    for images, labels in dataload:
        N = images.shape[0]
        x = images.reshape(N, -1)
        t = labels
        # loss
        loss = model.forward(x, t)
        total_loss += float(loss) * N
        # predictions
        probs = model.forward(x, t=None)
        preds = np.argmax(probs, axis=1)
        # ensure labels are integer class ids
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


def train(model, dataloader):
    batch_size = 128
    learning_rate = 0.01
    epochs = 100

    dataload = Dataloader(dataloader, is_train=True, shuffle=True, batch_size=batch_size)
    optimizer = SGD(learning_rate=learning_rate)

    out_dir = "checkpoints"
    os.makedirs(out_dir, exist_ok=True)
    train_losses = []
    test_losses  = []
    testload = Dataloader(dataloader, is_train=False, shuffle=False, batch_size=batch_size)

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
        train_losses.append(float(avg_loss))
        test_loss, test_acc = evaluate_loss_and_acc(model, testload)
        test_losses.append(float(test_loss))
        print(f"Epoch [{epoch+1}/{epochs}] completed. Train Loss: {avg_loss:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    with open(os.path.join(out_dir, "nn_loss_history.json"), "w") as f:
        json.dump({"train_losses": train_losses, "test_losses": test_losses}, f, indent=2)

if __name__ == "__main__":
    input_size = 28 * 28
    hidden_size1 = 128
    hidden_size2 = 64
    output_size = 10

    model = ThreeLayerNN(input_size, hidden_size1, hidden_size2, output_size)

    train(model, dataloader='dataset')
    save_model(model, save_path='checkpoints/nn.npz')
