import torch
from typing import List, Tuple

from pong.models import DQN
from pong.env import Reward, State, Action

class QLearningAgent:
    """
    Implements a Q-learning agent using a deep Q-network (DQN).
    """

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
        epsilon_decrease: float=0.1,
        min_epsilon: float=0.1,
        learning_rate: float=0.001,
        use_cuda: bool=False
    ) -> None:
        """
        Initialize the Q-learning agent.

        Args:
            q_function (DQN): The deep Q-network model.
            gamma (float): Discount factor for future rewards.
            epsilon (float): Exploration rate.
            epsilon_decrease (float): Rate at which epsilon decreases.
            min_epsilon (float): Minimum exploration rate.
            learning_rate (float): Learning rate for the optimizer.
            use_cuda (bool): Whether to use CUDA for computations.
        """

        self.gamma = gamma
        self.q_function = q_function

        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decrease = epsilon_decrease

        self.use_cuda = use_cuda
        self.loss_function = torch.nn.MSELoss(reduction="sum")
        self.loss: torch.Tensor | None = None
        self.nb_steps = 0
        self.total_nb_steps = 0

        if use_cuda:
            q_function = q_function.cuda()

        self.optimizer = torch.optim.Adam(self.q_function.parameters(), lr=learning_rate)

    def zero_loss(self) -> torch.Tensor | None:
        """
        Compute and return the average loss for the current step batch.
        Resets loss and step counters.

        Returns:
            torch.Tensor | None: The average loss, or None if no loss exists.
        """

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
        """
        Select an action based on the current policy (epsilon-greedy).

        Args:
            state (State): The current environment state.

        Returns:
            Action: The selected action.
        """

        if self.epsilon > torch.rand(1):
            return int(torch.randint(0, 6, (1,)))
        
        if self.use_cuda:
            state = state.cuda()

        return self.__select_action(state)

    def backward(
        self,
        batch: List[Tuple[State, Action, Reward, State]]
    ) -> None:
        """
        Perform a training step on a batch of transitions.

        Args:
            batch (List[Tuple[State, Action, Reward, State]]): 
                A batch of transitions (state, action, reward, next_state).
        """

        if self.total_nb_steps % 1000 == 0 and self.total_nb_steps != 0:
            self.epsilon = max(self.min_epsilon, self.epsilon - self.epsilon_decrease)

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
        """
        Compute the maximum predicted reward for a given state.

        Args:
            state (State): The input state.

        Returns:
            Reward: The maximum predicted reward.
        """

        out, _ = torch.max(self.q_function(state), -1, keepdim=True)
        return out.transpose(0, -1)

    def __select_action(
        self,
        state: State
    ) -> Action:
        """
        Select the optimal action for a given state based on the Q-function.

        Args:
            state (State): The input state.

        Returns:
            Action: The optimal action.
        """

        return int(torch.argmax(self.q_function(state)))

    def __preprocess_batch(
        self,
        batch: List[Tuple[State, Action, Reward, State]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess a batch of transitions for training.

        Args:
            batch (List[Tuple[State, Action, Reward, State]]): 
                A batch of transitions (state, action, reward, next_state).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The target values (Y) and 
            predicted values (Z) for the batch.
        """
        
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

        return Y, Z # type: ignore
