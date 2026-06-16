import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super().__init__()

        self.net = nn.Sequential(nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_dim))

        self.log_std = nn.Parameter(torch.zeros(action_dim)* 0.5)

    def forward(self, state):
        mean = self.net(state)
        log_std = torch.clamp(self.log_std, -5, 2)
        std = torch.exp(log_std)

        return mean, std
    

class Baseline(nn.Module):
    def __init__(self, state_dim, hidden_size):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, hidden_size)   
        self.fc2 = nn.Linear(hidden_size, 1)   

    def forward(self, state):
        hidden = F.relu(self.fc1(state))
        value = self.fc2(hidden)
        return value
    