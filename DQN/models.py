import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, state_channels, n_actions) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(state_channels, 16, 8, stride=4)  # Cx84x84
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2)  # Cx
        self.linear1 = nn.Linear(32 * 9 * 9, 256)
        self.linear2 = nn.Linear(256, n_actions)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.flatten(1)
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x
