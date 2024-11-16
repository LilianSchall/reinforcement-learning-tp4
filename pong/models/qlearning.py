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
        decreasing_rate: float=0.1,
        use_cuda: bool=False
    ) -> None:
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_function = q_function
        self.use_cuda = use_cuda
        self.loss_function = torch.nn.MSELoss(reduction="sum")
        self.loss: torch.Tensor | None = None
        self.nb_steps = 0
        self.total_nb_steps = 0
        self.decreasing_rate = decreasing_rate

        if use_cuda:
            q_function = q_function.cuda()

        self.optimizer = torch.optim.Adam(self.q_function.parameters(), lr=learning_rate)

    def zero_loss(self) -> torch.Tensor | None:
        if self.loss is None:
            return None
        avg_loss = self.loss.sum() / self.nb_steps
        self.loss = None
        self.nb_steps = 0
        return avg_loss

    def forward(
        self,
        state: State
    ) -> Action:
        if self.epsilon > torch.rand(1):
            return int(torch.randint(0, 6, (1,)))
        
        if self.use_cuda:
            state = state.cuda()

        return self.__select_action(state)

    def backward(
        self,
        batch: List[Tuple[State, Action, Reward, State]]
    ) -> None:
        if self.total_nb_steps % 1000 == 0 and self.total_nb_steps != 0:
            self.epsilon = max(0.1, self.epsilon - self.decreasing_rate)

        self.optimizer.zero_grad()
        
        Y, Z = self.__preprocess_batch(batch)
        loss = self.loss_function(Z, Y)
        loss.backward()

        self.optimizer.step()

        if self.loss is None:
            self.loss = loss
        else:
            self.loss += loss
        self.nb_steps += 1
        self.total_nb_steps += 1

    def __select_reward(
        self,
        state: State
    ) -> Reward:
        out, _ = torch.max(self.q_function(state), -1, keepdim=True)
        return out.transpose(0, -1)

    def __select_action(
        self,
        state: State
    ) -> Action:
        return int(torch.argmax(self.q_function(state)))

    def __preprocess_batch(
        self,
        batch: List[Tuple[State, Action, Reward, State]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        states, actions, rewards, next_states = zip(*batch)

        t_states = torch.vstack(states)
        t_next_states = torch.vstack(next_states)
        t_rewards = torch.tensor(rewards)

        if self.use_cuda:
            t_states = t_states.cuda()
            t_next_states = t_next_states.cuda()
            t_rewards = t_rewards.cuda()

        Z = self.__select_reward(t_states)
        Y = t_rewards + self.gamma * self.__select_reward(t_next_states) # type: ignore

        return Y, Z
