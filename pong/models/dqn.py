import torch
import torch.nn as nn

from pong.env import State

class DQN(nn.Module):
    def __init__(
        self,
        nb_actions: int
    ) -> None:
        pass

    def forward(
        self,
        x: State
    ) -> torch.Tensor:
        pass
