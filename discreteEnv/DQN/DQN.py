import random
from collections import deque

import gym
import numpy as np
import pygame
from gym import spaces
from keras import layers, optimizers
from tensorflow import keras


class DQNAgent:
    def __init__(self, state_shape, action_space, learning_rate=0.001):
        self.state_shape = state_shape
        self.action_space = action_space
        self.optimizer = optimizers.Adam(learning_rate)
        self.model = self.create_model()
        self.epsilon = 150.0
        self.min_epsilon = 0.01
        self.epsilon_decay = 0.995

    def create_model(self):
        model = keras.Sequential([
            layers.Input(shape=self.state_shape),
            layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.Flatten(),
            layers.Dense(24, activation='relu'),
            layers.Dense(self.action_space, activation='linear')
        ])
        model.compile(optimizer=self.optimizer, loss='mse')
        return model

    def choose_action(self, state):
        # Exploration
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        # Exploitation
        qs = self.model.predict(state.reshape(1, *self.state_shape))
        return np.argmax(qs[0])

    def train(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = reward + 0.99 * np.max(self.model.predict(next_state.reshape(1, *self.state_shape))[0])
        target_f = self.model.predict(state.reshape(1, *self.state_shape))
        target_f[0][action] = target
        self.model.fit(state.reshape(1, *self.state_shape), target_f, epochs=1, verbose=0)
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
        reward = 1 if self.grid[new_position] == 0 else -100
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

        self.screen.fill((255, 255, 255))  # White background
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
state_shape = env.observation_space.shape + (1,)
agent = DQNAgent(state_shape=state_shape, action_space=env.action_space.n)
replay_buffer = ReplayBuffer(10000)


def train_dqn(episodes):
    for episode in range(episodes):
        state = env.reset().reshape(*state_shape)
        done = False
        total_reward = 0
        while not done:
            env.render()
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.reshape(*state_shape)
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if len(replay_buffer.buffer) > 500:
                batch = replay_buffer.sample(32)
                for s, a, r, s_next, d in batch:
                    agent.train(s, a, r, s_next, d)
            if done:
                print(f"Episode {episode + 1}: Total reward = {total_reward}, Epsilon = {agent.epsilon:.2f}")
                break

train_dqn(5)
env.close()
