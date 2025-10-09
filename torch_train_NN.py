from model.torch_ThreeLayerNN import ThreeLayerNN
import os, json
import torch
import torch.nn as nn
import torch.optim as optim
from src.dataloader import Dataloader
import numpy as np

def np_batch_to_tensors(images, labels, device):
    if isinstance(images, np.ndarray):
        x = torch.from_numpy(images).float()
    else:
        x = images.float() if isinstance(images, torch.Tensor) else images
    if hasattr(x, 'ndim') and x.ndim == 3:
        x = x.unsqueeze(1)

    if isinstance(labels, np.ndarray):
        y_np = labels.argmax(axis=1) if labels.ndim == 2 else labels
        y = torch.from_numpy(y_np).long()
    else:
        y = labels.long() if isinstance(labels, torch.Tensor) else labels
    return x.to(device), y.to(device)

@torch.no_grad()
def evaluate_loss_and_acc(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = np_batch_to_tensors(images, labels, device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total += images.size(0)
    return total_loss / total, total_correct / total


def save_model(model, save_path='checkpoints/nn.pt'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model parameters saved to {save_path}")

def train(model, dataloader, device):
    # Hyperparameters
    batch_size = 128
    learning_rate = 0.01
    epochs = 5

    dataload = Dataloader(dataloader, is_train=True, shuffle=True, batch_size=batch_size)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    out_dir = "checkpoints"
    os.makedirs(out_dir, exist_ok=True)
    train_losses = []
    test_losses  = []
    testload = Dataloader(dataloader, is_train=False, shuffle=False, batch_size=batch_size)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        seen = 0
        for images, labels in dataload:
            images, labels = np_batch_to_tensors(images, labels, device)  # labels: int64 class ids
            optimizer.zero_grad()
            logits = model(images)
            loss = nn.CrossEntropyLoss()(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            seen += images.size(0)


        avg_train_loss = total_loss / seen
        train_losses.append(avg_train_loss)

        avg_test_loss, test_acc = evaluate_loss_and_acc(model, testload, nn.CrossEntropyLoss(), device)
        test_losses.append(avg_test_loss)

        print(f"Epoch {epoch+1}/{epochs} complemted. Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | Test Acc: {test_acc:.4f}")

    with open(os.path.join(out_dir, 'epoch_5_torch_nn_loss_history.json'), 'w') as f:
        json.dump({'train_losses': train_losses, 'test_losses': test_losses}, f, indent=2)
    
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ThreeLayerNN(input_size=28*28, hidden_size1=128, hidden_size2=64, output_size=10)
    train(model.to(device), dataloader='dataset', device=device)

    save_model(model, save_path='checkpoints/epoch_5_torch_nn.pt')
    