import gym
from gym import spaces
import pygame
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        self.size = int(size)  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        self.step_count = 0
        self.max_steps = 100  # maximum number of steps before episode gets terminated

        # Observations are a flat array consisting of the grid and agent's location.
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.size * self.size + 2,), dtype=int
        )

        # We have 5 actions, corresponding to "right", "up", "left", "down", and "stop"
        self.action_space = spaces.Discrete(5)

        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
            4: "Finished"  # any string works
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def _get_obs(self):
        grid_flat = self.grid.flatten()
        agent_location = np.array(self._agent_location)
        return np.concatenate([grid_flat, agent_location])

    def _get_info(self):
        return {
            "percentage covered": (np.sum(self.grid) / self.size ** 2 * 100),
            "steps taken": self.step_count
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0
        self.reward = 0

        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        self.grid = np.zeros([self.size, self.size], dtype=int)
        self.grid[tuple(self._agent_location)] = 1

        self.render_grid = self.grid.copy()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        self.step_count += 1
        direction = self._action_to_direction[action]

        if isinstance(direction, str):  # Finished action
            terminated = True
        else:
            old_location = self._agent_location.copy()
            self._agent_location = np.clip(self._agent_location + direction, 0, self.size - 1)

            if np.array_equal(old_location, self._agent_location):
                self.reward = -1  # penalty for trying to move out of bounds
                self.render_grid[tuple(self._agent_location)] += 1
            elif self.grid[tuple(self._agent_location)] == 0:
                self.reward = 10  # reward for covering new square
                self.grid[tuple(self._agent_location)] = 1
                self.render_grid[tuple(self._agent_location)] += 1
            else:
                self.reward = -1  # penalty for revisiting square
                self.render_grid[tuple(self._agent_location)] += 1

            terminated = np.all(self.grid != 0)

        truncated = self.step_count == self.max_steps

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, self.reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (self.window_size / self.size)

        cell_width = self.window_size // self.size
        cell_height = self.window_size // self.size

        for i in range(self.size):
            for j in range(self.size):
                if self.render_grid[j, i] == 0:
                    color = (247, 247, 247)
                else:
                    color = (max(31 - self.render_grid[j, i] * 20, 0), max(243 - self.render_grid[j, i] * 20, 0),
                             max(90 - self.render_grid[j, i] * 20, 0))
                rect = pygame.Rect(j * cell_width, i * cell_height, cell_width, cell_height)
                pygame.draw.rect(canvas, color, rect)

        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
# Create the environment
env = GridWorldEnv()

# Create the model
model = DQN("MlpPolicy", env, verbose=1, learning_rate=0.0003)

# Train the agent
checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./logs/', name_prefix='dqn_model')
total_timesteps = 1000000
model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback])

# Evaluate the trained agent
eval_env = GridWorldEnv(render_mode="human")
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, render=True)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

eval_env.close()
