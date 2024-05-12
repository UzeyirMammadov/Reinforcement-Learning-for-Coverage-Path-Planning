import os
from random import random

import numpy as np
import torch
from torch import optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from discreteEnv.DQN.DQN import DQN, agent


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
        agent.writer.add_scalar('Average Q-value', average_q_value)
        loss = self.loss_fn(self.model(state), target_f)
        loss.backward()
        self.optimizer.step()

        if done:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)