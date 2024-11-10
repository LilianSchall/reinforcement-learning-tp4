from typing import SupportsFloat, Tuple
import gymnasium as gym
import numpy as np
import ale_py
import torch
from torchvision import transforms

State = torch.Tensor
Action = int
Reward = float | SupportsFloat

class PongEnvironment:
    
    loaded_gym = False
    action_space: int
    frame_size: int

    def __init__(self) -> None:
        if not PongEnvironment.loaded_gym:
            gym.register_envs(ale_py)

        self.env = gym.make("ALE/Pong-v5", obs_type="grayscale")
        self.action_space = self.env.action_space.n # type: ignore
        self.frame_size = 88

        self.frame_preprocessing = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((100, 84)),
            transforms.CenterCrop(self.frame_size)
        ])

    def reset(self) -> State:
        state, _ = self.env.reset()
        return self.__process_gym_state(state)

    def get_nb_actions(self) -> int:
        return self.action_space

    def step(self, action: Action) -> Tuple[State, Reward, bool]:
        next_state, reward, done, _, _ = self.env.step(action)
        return (
            self.__process_gym_state(next_state),
            reward,
            done
        )

    def __process_gym_state(self, state: np.ndarray) -> State:
        state = state[...,None]
        return self.frame_preprocessing(state) # type: ignore
        
