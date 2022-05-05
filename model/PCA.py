import torch.nn as nn


class PCA(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PCA, self).__init__()
        self.model_type = 'PCA'
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
