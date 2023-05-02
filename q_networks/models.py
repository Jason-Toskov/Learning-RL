import torch
import torch.nn as nn
from torch.nn import functional as F

class QNetwork(nn.Module):
    def __init__(self, state_channels, n_actions) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(state_channels, 32, kernel_size=8, stride=4) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2) 
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.linear1 = nn.Linear(3136, 512)
        self.linear2 = nn.Linear(512, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
