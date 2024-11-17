from os.path import isdir
from typing import List, Tuple
from pong.models import DQN, QLearningAgent
from pong.env import PongEnvironment, State, Action, Reward

from tqdm import tqdm

import torch
import numpy as np
import random
import gc
import os

class Session:
    
    environment: PongEnvironment
    agent: QLearningAgent
    memory: List[Tuple[State, Action, Reward, State]]

    def __init__(
        self,
        nb_epochs: int,
        max_nb_steps: int,
        memory_size: int,
        batch_size: int,
        training: bool,
        load_checkpoint: str | None=None,
        checkpoint_frequency: int | None=None,
    ):
        self.nb_epochs = nb_epochs
        self.max_nb_steps = max_nb_steps

        self.memory_size = memory_size
        self.memory = []

        self.batch_size = batch_size
        self.training = training


        self.use_cuda = torch.cuda.is_available()

        self.environment = PongEnvironment(with_video=True)
        q_function = DQN(self.environment.get_nb_actions())

        if load_checkpoint is not None:
            q_function.load_state_dict(torch.load(load_checkpoint, weights_only=True))
            print("Loaded model from: " + load_checkpoint)

        self.agent = QLearningAgent(
            q_function,
            0.99,
            1.0,
            0.001,
            0.1,
            0.0001,
            self.use_cuda
        )

        self.checkpoint_frequency = checkpoint_frequency
        self.file_path_base = "model_save/dqn-epoch" # end with .pth

        if not os.path.isdir("model_save"):
            os.mkdir("model_save")
        

    def run(self):
        scores: List[Reward] = []
        for epoch in range(self.nb_epochs):
            gc.collect()
            if self.use_cuda:
                torch.cuda.empty_cache()
            
            current_state: State = self.environment.reset()
            print(f"Epoch {epoch}")
            rewards = []
            self.agent.zero_loss()
            action = 0
            score = 0
            for step in tqdm(range(self.max_nb_steps)):
                action = self.agent.forward(current_state)
                next_state, reward, done = self.environment.step(action)

                rewards.append(reward)
                if len(self.memory) >= self.memory_size:
                    self.memory.pop(0)
                self.memory.append((current_state, action, reward, next_state))

                self.agent.backward(
                    random.choices(
                        self.memory,
                        k=min(self.batch_size, len(self.memory))
                    )
                )
                if done:
                    break
                current_state = next_state
                score += reward # type: ignore

            scores.append(score)
            loss = self.agent.zero_loss()

            print(f"Average score: {np.mean(scores[-100:])}")
            print(f"last score: {score}")
            print(f"Average loss: {loss}")
            print(f"Current epsilon: {self.agent.epsilon}")

            if self.training and\
                    self.checkpoint_frequency is not None and\
                    epoch % self.checkpoint_frequency == 0:
                model_path = self.file_path_base + f"_{epoch}.pth"
                torch.save(
                    self.agent.q_function.state_dict(), 
                    model_path
                )
                print("Saved model to: " + model_path)

    def close(self):
        self.environment.close()
