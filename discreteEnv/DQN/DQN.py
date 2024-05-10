import random
from collections import deque

import gym
import numpy as np
import pygame
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from tensorflow.python.training.training_util import global_step
from torch.utils.tensorboard import SummaryWriter
import os
import time

class DQN(nn.Module):
    def __init__(self, state_shape, action_space):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(state_shape[0], 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * state_shape[1] * state_shape[2], 24)
        self.fc2 = nn.Linear(24, action_space)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class DQNAgent:
    def __init__(self, state_shape, action_space, learning_rate=0.001):
        self.state_shape = state_shape
        self.action_space = action_space
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_shape, action_space).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.epsilon = 1.0
        self.min_epsilon = 0.01
        self.epsilon_decay = 0.995
        self.loss_fn = nn.MSELoss()

        log_dir = os.path.join("logs", "dqn")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.writer = SummaryWriter(log_dir=log_dir)

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        state = torch.tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def train(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device).unsqueeze(0)
        reward = torch.tensor(reward, dtype=torch.float32).to(self.device)
        action = torch.tensor(action, dtype=torch.int64).to(self.device)
        done = torch.tensor(done, dtype=torch.float32).to(self.device)

        target = reward
        if not done:
            target = reward + 0.99 * torch.max(self.model(next_state)).item()
        target_f = self.model(state)
        target_f[0][action] = target

        self.optimizer.zero_grad()
        average_q_value = torch.mean(target_f).item()
        agent.writer.add_scalar('Average Q-value', average_q_value, global_step)
        loss = self.loss_fn(self.model(state), target_f)
        loss.backward()
        self.optimizer.step()

        if done:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

class CoverageEnv(gym.Env):
    def __init__(self, grid_size=(5, 5), cell_size=50):
        super(CoverageEnv, self).__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=1, shape=grid_size, dtype=np.int32)
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.screen_size = (grid_size[0] * cell_size, grid_size[1] * cell_size)
        self.grid = np.zeros(grid_size, dtype=np.int32)
        self.agent_position = [0, 0]
        pygame.init()
        self.screen = pygame.display.set_mode(self.screen_size)
        pygame.display.set_caption("Coverage Environment")
        self.clock = pygame.time.Clock()

    def step(self, action):
        prev_position = tuple(self.agent_position)
        if action == 0:
            self.agent_position[0] = max(self.agent_position[0] - 1, 0)
        elif action == 1:
            self.agent_position[0] = min(self.agent_position[0] + 1, self.grid_size[0] - 1)
        elif action == 2:
            self.agent_position[1] = max(self.agent_position[1] - 1, 0)
        elif action == 3:
            self.agent_position[1] = min(self.agent_position[1] + 1, self.grid_size[1] - 1)

        new_position = tuple(self.agent_position)
        reward = 1 if self.grid[new_position] == 0 else -1
        self.grid[new_position] = 1
        done = np.all(self.grid == 1)
        if done:
            reward += 10

        return self.grid.copy(), reward, done, {}

    def reset(self):
        self.grid = np.zeros(self.grid_size, dtype=np.int32)
        self.agent_position = [0, 0]
        return self.grid.copy()

    def render(self, mode='human', close=False):
        if close:
            pygame.quit()
            return

        self.screen.fill((255, 255, 255))
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                rect = pygame.Rect(y * self.cell_size, x * self.cell_size, self.cell_size, self.cell_size)
                color = (0, 255, 0) if self.grid[x, y] == 1 else (255, 255, 255)
                pygame.draw.rect(self.screen, color, rect, 0)
                pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)

        agent_rect = pygame.Rect(self.agent_position[1] * self.cell_size, self.agent_position[0] * self.cell_size,
                                 self.cell_size, self.cell_size)
        pygame.draw.ellipse(self.screen, (255, 0, 0), agent_rect)
        pygame.display.flip()
        self.clock.tick(60)

env = CoverageEnv(grid_size=(5, 5))
state_shape = (1,) + env.observation_space.shape
agent = DQNAgent(state_shape=state_shape, action_space=env.action_space.n)
replay_buffer = ReplayBuffer(20000)

def train_dqn(episodes):
    for episode in range(episodes):
        state = env.reset().reshape(state_shape)
        done = False
        total_reward = 0
        while not done:
            env.render()
            time.sleep(0.1)
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.reshape(state_shape)  # Adjust reshaping
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if len(replay_buffer.buffer) > 500:
                batch = replay_buffer.sample(32)
                for s, a, r, s_next, d in batch:
                    agent.train(s, a, r, s_next, d)
                    loss = agent.loss_fn(
                        agent.model(torch.tensor(s, dtype=torch.float32).to(agent.device).unsqueeze(0)),
                        agent.model(torch.tensor(s, dtype=torch.float32).to(agent.device).unsqueeze(0)))
                    agent.writer.add_scalar('Loss', loss.item(), global_step)
            if done:
                print(f"Episode {episode + 1}: Total reward = {total_reward}, Epsilon = {agent.epsilon:.2f}")

            agent.writer.add_scalar('Epsilon', agent.epsilon, episode)
            agent.writer.add_scalar('Total Reward', total_reward, episode)
            agent.writer.add_scalar('Buffer Size', len(replay_buffer.buffer), episode)

    env.close()

train_dqn(20)
