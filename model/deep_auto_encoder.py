import torch.nn as nn
import torch.nn.functional


class DAE(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(DAE, self).__init__()
        self.model_type = 'dae'
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2 + 1)
        self.fc3 = nn.Linear(hidden_size // 2 + 1, hidden_size)
        self.fc4 = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.leaky_relu(x)
        x = self.fc2(x)
        x = nn.functional.leaky_relu(x)
        x = self.fc3(x)
        x = nn.functional.leaky_relu(x)
        x = self.fc4(x)
        return x
