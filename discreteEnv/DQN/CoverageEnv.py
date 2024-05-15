import numpy as np
import pygame
import gym
from gym import spaces
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

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
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