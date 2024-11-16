from typing import SupportsFloat, Tuple
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
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
    frame_buffer: list[State]

    def __init__(self, with_video=False) -> None:
        if not PongEnvironment.loaded_gym:
            gym.register_envs(ale_py)
            PongEnvironment.loaded_gym = True

        self.env = gym.make("ALE/Pong-v5", obs_type="grayscale", render_mode="rgb_array")
        
        if with_video:
            self.env = RecordVideo(
                self.env,
                video_folder="videos",
                name_prefix="training",
                episode_trigger=lambda x: True
            )

        self.action_space = self.env.action_space.n # type: ignore
        self.frame_size = 84

        self.frame_preprocessing = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((84, 110)),
            transforms.CenterCrop(self.frame_size),
            lambda x: x>0.5,
            lambda x: x.float(),
        ])
        self.frame_buffer = []

    def reset(self) -> State:
        state, _ = self.env.reset()
        frame: State =  self.__process_gym_state(state)
        self.frame_buffer = [frame.clone()] * 4
        returned_state: State = self.__produce_concatenated_state()
        return returned_state

    def close(self):
        self.env.close()

    def get_nb_actions(self) -> int:
        return self.action_space

    def step(self, action: Action) -> Tuple[State, Reward, bool]:
        next_state, reward, done, _, _ = self.env.step(action)
        t = (
            self.__produce_concatenated_state(self.__process_gym_state(next_state)),
            reward,
            done
        )
        return t

    def __process_gym_state(self, state: np.ndarray) -> State:
        state = state.T[...,None]
        return self.frame_preprocessing(state) # type: ignore

    def __produce_concatenated_state(self, new_state=None) -> State:
        if new_state is not None:
            self.frame_buffer.pop(0)
            self.frame_buffer.append(new_state)

        returned_state: State = torch.vstack(self.frame_buffer)
        returned_state = returned_state[None, ...]
        return returned_state
