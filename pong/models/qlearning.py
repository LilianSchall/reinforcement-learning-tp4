import torch
from typing import List, Tuple

class QLearningAgent:
    def __init__(
        self, 
        gamma: float,
    ) -> None:
        pass

    def forward(
        self,
        state: torch.Tensor
    ) -> int:
        return 0

    def backward(
        self,
        batch: List[Tuple[torch.Tensor, int, float, torch.Tensor]]
    ) -> None:
        pass
