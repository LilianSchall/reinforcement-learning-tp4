import torch
import torch.nn as nn
import torch.nn.functional as F

from pong.env import State

class DQN(nn.Module):
    """
    Deep Q-Network (DQN) implementation using convolutional and fully connected layers.
    """
    def __init__(
        self,
        nb_actions: int
    ) -> None:
        """
        Initialize the DQN model with convolutional and fully connected layers.

        Args:
            nb_actions (int): Number of possible actions the agent can take.
        """
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, nb_actions)

    def forward(
        self,
        x: State
    ) -> torch.Tensor:
        """
        Perform a forward pass through the network.

        Args:
            x (State): Input state tensor, typically a stack of consecutive frames.

        Returns:
            torch.Tensor: Q-values for each possible action.
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
