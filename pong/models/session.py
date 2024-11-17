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

Memory = List[Tuple[State, Action, Reward, State]]

class Session:
    environment: PongEnvironment
    agent: QLearningAgent

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

        self.batch_size = batch_size
        self.training = training


        self.use_cuda = torch.cuda.is_available()

        self.environment = PongEnvironment(with_video=True)
        q_function = DQN(self.environment.get_nb_actions())

        if load_checkpoint is not None:
            q_function.load_state_dict(torch.load(load_checkpoint, weights_only=True))
            print("Loaded model from: " + load_checkpoint)

        self.agent = QLearningAgent(
            q_function=q_function,
            gamma=0.99,
            epsilon=1.0 if training else 0.1,
            epsilon_decrease=0.001 if training else 0,
            min_epsilon=0.1,
            learning_rate=0.0001,
            use_cuda=self.use_cuda
        )

        self.checkpoint_frequency = checkpoint_frequency
        self.file_path_base = "model_save/dqn-epoch" # end with .pth

        if not os.path.isdir("model_save"):
            os.mkdir("model_save")
        

    def run(self):
        scores: List[Reward] = []
        memory: Memory = []
        
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
                if self.training:
                    self.__update_agent(
                        memory,
                        current_state,
                        action,
                        reward,
                        next_state
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

    def __update_agent(
        self, 
        memory: Memory,
        current_state: State,
        action: Action,
        reward: Reward,
        next_state: State
    ):
        if len(memory) >= self.memory_size:
            memory.pop(0)
        memory.append((current_state, action, reward, next_state))
        
        self.agent.backward(
            random.choices(
                memory,
                k=min(self.batch_size, len(memory))
            )
        )

    def close(self):
        self.environment.close()
