from typing import List, Tuple
from pong.models import DQN, QLearningAgent
from pong.env import PongEnvironment, State, Action, Reward

import torch

def main(nb_epochs: int, max_nb_steps: int, max_memory: int):
    environment = PongEnvironment()
    q_function  = DQN(environment.get_nb_actions())
    agent       = QLearningAgent(q_function, 0.99, 0.01)
    memory: List[Tuple[State, Action, Reward, State]] = []

    current_state: State = environment.reset()

    for epoch in range(nb_epochs):
        for step in range(max_nb_steps):
            
            action = agent.forward(current_state)
            next_state, reward, done = environment.step(action)

            if len(memory) >= max_memory:
                memory.pop()
            memory.append((current_state, action, reward, next_state))

            # For the moment, we give the entire memory
            # We will think about how building the mini batch later.
            # Maybe using a DataLoader?
            agent.backward(memory)
            if done:
                break



if __name__ == "__main__":
    main(100, 1000, 500)

