import numpy as np
from src.dataloader import Dataloader
from src.SGD import SGD
import os
import json

from model.ThreeLayerCNN import ThreeLayerCNN


def evaluate_loss_and_acc(model, dataload):
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

def save_model(model, save_path='checkpoints/cnn.npz'):
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



def train(model, dataloader):
    batch_size = 128
    learning_rate = 0.01
    epochs = 10 
    
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
            
            x_batch = images
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

    with open(os.path.join(out_dir, "cnn_loss_history.json"), "w") as f:
        json.dump({"train_losses": train_losses, "test_losses": test_losses}, f, indent=2)

if __name__ == "__main__":
    input_dim = (1, 28, 28)
    output_size = 10

    model = ThreeLayerCNN(input_dim=input_dim, output_size=output_size)

    train(model, dataloader='dataset') 
    save_model(model, save_path='checkpoints/cnn.npz')