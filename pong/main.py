from typing import List, Tuple
from pong.models import DQN, QLearningAgent
from pong.env import PongEnvironment, State, Action, Reward

from tqdm import tqdm

import torch
import random

def main(nb_epochs: int, max_nb_steps: int, max_memory: int, batch_size: int):
    environment = PongEnvironment(with_video=True)
    q_function  = DQN(environment.get_nb_actions())
    agent       = QLearningAgent(q_function, 0.99, 1.0, 0.01, 0.05)
    memory: List[Tuple[State, Action, Reward, State]] = []

    for epoch in range(nb_epochs):
        current_state: State = environment.reset()
        print(f"Epoch {epoch}")
        rewards = []
        agent.zero_loss()

        for step in tqdm(range(max_nb_steps)):
            action = agent.forward(current_state)
            next_state, reward, done = environment.step(action)

            rewards.append(reward)
            if len(memory) >= max_memory:
                memory.pop(0)
            memory.append((current_state, action, reward, next_state))

            agent.backward(random.choices(memory, k=min(batch_size, len(memory))))
            if done:
                break

        loss = agent.zero_loss()
        nb_steps_needed = len(rewards)
        mean_reward = sum(rewards) / nb_steps_needed 
        print(f"Nb steps needed for epoch: {nb_steps_needed}")
        print(f"Average reward: {mean_reward}")
        print(f"min reward: {min(rewards)}")
        print(f"max reward: {max(rewards)}")
        print(f"Average loss: {loss}")
    environment.close()



if __name__ == "__main__":
    main(20, 1000, 500, 64)

