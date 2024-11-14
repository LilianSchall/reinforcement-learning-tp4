import torch
from typing import List, Tuple

from pong.models import DQN
from pong.env import Reward, State, Action

class QLearningAgent:

    gamma:      float
    epsilon:    float
    q_function: DQN
    optimizer: torch.optim.Optimizer
    loss_function: torch.nn.MSELoss

    def __init__(
        self, 
        q_function:    DQN,
        gamma:         float,
        epsilon:       float,
        learning_rate: float=0.001,
    ) -> None:
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_function = q_function
        self.optimizer = torch.optim.Adam(self.q_function.parameters(), lr=learning_rate)
        self.loss_function = torch.nn.MSELoss()

    def forward(
        self,
        state: State
    ) -> Action:
        if self.epsilon > torch.rand(1):
            return int(torch.randint(0, 6, (1,)))
        return self.__select_action(state)

    def backward(
        self,
        batch: List[Tuple[State, Action, Reward, State]]
    ) -> None:
        self.optimizer.zero_grad()
        
        Y, Z = self.__preprocess_batch(batch)
        loss = self.loss_function(Z, Y)
        loss.backward()

        self.optimizer.step()

    def __select_reward(
        self,
        state: State
    ) -> Reward:
        return torch.max(self.q_function(state))

    def __select_action(
        self,
        state: State
    ) -> Action:
        return int(torch.argmax(self.q_function(state)))

    def __compute_y(
        self,
        next_state: State,
        reward:     Reward,
    ) -> float:
        ## TODO: needs to include if next_state is terminal
        return reward + self.gamma * self.__select_reward(next_state) # type: ignore

    def __preprocess_batch(
        self,
        batch: List[Tuple[State, Action, Reward, State]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ys = []
        zs = []
        for transition in batch:
            ys.append(self.__compute_y(transition[3], transition[2]))
            zs.append(self.__select_reward(transition[0]))

        Y = torch.tensor(ys, requires_grad=True)
        Z = torch.tensor(zs, requires_grad=True)
        return Y, Z
