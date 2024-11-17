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
    """
    Represents the Pong environment for reinforcement learning using OpenAI Gym.
    Manages state processing, environment interaction, and action space config.
    """
    
    loaded_gym = False
    action_space: int
    frame_size: int
    frame_buffer: list[State]

    def __init__(self, with_video=False) -> None:
        """
        Initializes the Pong environment.

        Args:
            with_video (bool, optional): If True, records gameplay as videos.
            Defaults to False.
        """
        if not PongEnvironment.loaded_gym:
            gym.register_envs(ale_py)
            PongEnvironment.loaded_gym = True

        self.env = gym.make(
            "ALE/Pong-v5",
            obs_type="grayscale",
            render_mode="rgb_array"
        )
        
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
        """
        Resets the environment to its initial state.

        Returns:
            State: The initial state of the environment, preprocessed.
        """
        state, _ = self.env.reset()
        frame: State =  self.__process_gym_state(state)
        self.frame_buffer = [frame.clone()] * 4
        returned_state: State = self.__produce_concatenated_state()
        return returned_state

    def close(self):
        """
        Closes the environment and releases resources.
        """
        self.env.close()

    def get_nb_actions(self) -> int:
        """
        Returns the number of possible actions in the action space.

        Returns:
            int: Number of possible actions.
        """
        return self.action_space

    def step(self, action: Action) -> Tuple[State, Reward, bool]:
        """
        Performs a step in the environment using the specified action.

        Args:
            action (Action): The action to execute.

        Returns:
            Tuple[State, Reward, bool]: A tuple containing the next state,
            the reward received, and a flag indicating if the episode has ended.
        """
        next_state, reward, done, _, _ = self.env.step(action)
        t = (
            self.__produce_concatenated_state(self.__process_gym_state(next_state)),
            reward,
            done
        )
        return t

    def __process_gym_state(self, state: np.ndarray) -> State:
        """
        Applies preprocessing to a raw Gym state.

        Args:
            state (np.ndarray): The raw state from the Gym environment.

        Returns:
            State: The preprocessed state as a tensor.
        """
        state = state.T[...,None]
        return self.frame_preprocessing(state) # type: ignore

    def __produce_concatenated_state(self, new_state=None) -> State:
        """
        Produces a concatenated state from the frame buffer.
        Updates the buffer if a new state is provided.

        Args:
            new_state (State, optional): A new state to add to the buffer.
            Defaults to None.

        Returns:
            State: The concatenated state tensor.
        """
        if new_state is not None:
            self.frame_buffer.pop(0)
            self.frame_buffer.append(new_state)

        returned_state: State = torch.vstack(self.frame_buffer)
        returned_state = returned_state[None, ...]
        return returned_state
