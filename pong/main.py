from typing import List, Tuple
from pong.models import DQN, QLearningAgent
from pong.env import PongEnvironment, State, Action, Reward

from tqdm import tqdm

import torch
import numpy as np
import random
import gc

def main(nb_epochs: int, max_nb_steps: int, max_memory: int, batch_size: int, use_cuda: bool):
    environment = PongEnvironment(with_video=True)
    q_function  = DQN(environment.get_nb_actions())
    agent       = QLearningAgent(q_function, 0.99, 1.0, 0.0001, 0.001, use_cuda)
    memory: List[Tuple[State, Action, Reward, State]] = []

    scores = []

    for epoch in range(nb_epochs):
        gc.collect()
        if use_cuda:
            torch.cuda.empty_cache()
        
        current_state: State = environment.reset()
        print(f"Epoch {epoch}")
        rewards = []
        agent.zero_loss()
        action = 0
        score = 0
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
            current_state = next_state
            score += reward # type: ignore

        scores.append(score)
        loss = agent.zero_loss()

        print(f"Average score: {np.mean(scores[-100:])}")
        print(f"last score: {score}")
        print(f"Average loss: {loss}")
        print(f"Current epsilon: {agent.epsilon}")
    environment.close()



if __name__ == "__main__":
    main(3000, 10000, 10000, 64, torch.cuda.is_available())

