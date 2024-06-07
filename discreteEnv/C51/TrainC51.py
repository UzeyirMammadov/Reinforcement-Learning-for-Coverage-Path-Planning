import os

import numpy as np
import torch

from discreteEnv.C51.C51Agent import C51Agent
from discreteEnv.C51.CoverageEnv import CoverageEnv

env = CoverageEnv(grid_size=(5, 5))
state_dim = np.prod(env.observation_space.shape)
action_dim = env.action_space.n

agent = C51Agent(state_dim, action_dim)

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)

def load_checkpoint(filename="checkpoint.pth.tar"):
    return torch.load(filename)

def train_c51(episodes, render=False):
    for episode in range(episodes):
        state = env.reset().flatten()
        episode_reward = 0
        done = False
        step_count = 0

        while not done:
            if render:
                env.render()
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.flatten()
            agent.add_experience(state, action, reward, next_state, done)
            agent.train()
            state = next_state
            episode_reward += reward
            step_count += 1
            if done:
                agent.epsilon = max(agent.min_epsilon, agent.epsilon * agent.epsilon_decay)
                print(f"Episode {episode + 1}: Total Reward = {episode_reward}, Steps = {step_count}, Epsilon = {agent.epsilon:.2f}")
                break

        checkpoint = {
            'episode': episode,
            'model_state_dict': agent.model.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'epsilon': agent.epsilon
        }
        save_checkpoint(checkpoint)


start_episode = 0
if os.path.exists("checkpoint.pth.tar"):
    checkpoint = load_checkpoint()
    agent.model.load_state_dict(checkpoint['model_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_episode = checkpoint['episode'] + 1
    agent.epsilon = checkpoint['epsilon']

print(f"Starting training from episode {start_episode}")
train_c51(100, render=True)
env.close()