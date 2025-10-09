from src.dataloader import Dataloader
from src.SGD import SGD
from src.utils import evaluate_loss_and_acc_cnn, save_model_cnn
import os
import json
from model.ThreeLayerCNN import ThreeLayerCNN

def train(model, dataloader):
    # Hyperparameters
    batch_size = 128
    learning_rate = 0.01
    epochs = 500
    
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
        
        test_loss, test_acc = evaluate_loss_and_acc_cnn(model, testload)
        test_losses.append(float(test_loss))
        
        print(f"Epoch [{epoch+1}/{epochs}] completed. Train Loss: {avg_loss:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    with open(os.path.join(out_dir, "epoch_500_cnn_loss_history.json"), "w") as f:
        json.dump({"train_losses": train_losses, "test_losses": test_losses}, f, indent=2)

if __name__ == "__main__":
    input_dim = (1, 28, 28)
    output_size = 10

    model = ThreeLayerCNN(input_dim=input_dim, output_size=output_size)

    train(model, dataloader='dataset') 
    save_model_cnn(model, save_path='checkpoints/epoch_500_cnn.npz')