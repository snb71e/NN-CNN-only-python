import torch
import torch.nn as nn
import torch.optim as optim

class ThreeLayerNN(nn.Module):
    def __init__(self, input_size=28*28, hidden_size1=128, hidden_size2=64, output_size=10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, output_size)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1) 
        return self.network(x)

    def probabilities(self, x):
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)