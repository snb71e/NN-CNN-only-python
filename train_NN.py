from src.dataloader import Dataloader
from src.SGD import SGD
from src.utils import evaluate_loss_and_acc, save_model
import os
import json
from model.ThreeLayerNN import ThreeLayerNN

def train(model, dataloader):
    # Hyperparameters
    batch_size = 128
    learning_rate = 0.01
    epochs = 5

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

    with open(os.path.join(out_dir, "epoch_5_nn_loss_history.json"), "w") as f:
        json.dump({"train_losses": train_losses, "test_losses": test_losses}, f, indent=2)

if __name__ == "__main__":
    input_size = 28 * 28
    hidden_size1 = 128
    hidden_size2 = 64
    output_size = 10

    model = ThreeLayerNN(input_size, hidden_size1, hidden_size2, output_size)

    train(model, dataloader='dataset')
    save_model(model, save_path='checkpoints/epoch_5_nn.npz')
