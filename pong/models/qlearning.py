import torch
from typing import List, Tuple

from pong.models import DQN
from pong.env import Reward, State, Action

class QLearningAgent:

    gamma: float
    q_function: DQN

    def __init__(
        self, 
        gamma: float,
        q_function: DQN
    ) -> None:
        self.gamma = gamma
        self.q_function = q_function

    def forward(
        self,
        state: State
    ) -> Action:
        return 0

    def backward(
        self,
        batch: List[Tuple[State, Action, Reward, State]]
    ) -> None:
        pass
