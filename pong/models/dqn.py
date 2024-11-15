import torch
import torch.nn as nn
import torch.nn.functional as F

from pong.env import State

class DQN(nn.Module):
    def __init__(
        self,
        nb_actions: int
    ) -> None:
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, 8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2)
        self.fc1 = nn.Linear(32 * 9 * 9, 256)
        self.fc2 = nn.Linear(256, nb_actions)

    def forward(
        self,
        x: State
    ) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
