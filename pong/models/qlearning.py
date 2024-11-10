import torch
from typing import List, Tuple

from pong.models import DQN
from pong.env import Reward, State, Action

class QLearningAgent:

    gamma: float
    epsilon: float
    q_function: DQN

    def __init__(
        self, 
        gamma: float,
        epsilon: float,
        q_function: DQN
    ) -> None:
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_function = q_function

    def forward(
        self,
        state: State
    ) -> Action:
        if self.epsilon > torch.rand(1):
            return int(torch.randint(0, 6, (1,)))
        return int(torch.argmax(self.q_function(state)))

    def backward(
        self,
        batch: List[Tuple[State, Action, Reward, State]]
    ) -> None:
        pass
