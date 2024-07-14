import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from shapely.geometry import Point, Polygon
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor


class ContinuousPolygonCoverageEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self, field_points, bounding_box_points, cell_size=1, max_steps=1000, coverage_threshold=0.95,
                 render_mode=None):
        super(ContinuousPolygonCoverageEnv, self).__init__()
        self.field_polygon = Polygon(field_points)
        self.bounding_box = Polygon(bounding_box_points)
        self.cell_size = cell_size
        self.max_steps = max_steps
        self.current_step = 0
        self.coverage_threshold = coverage_threshold
        self.render_mode = render_mode

        self.action_space = spaces.Box(low=np.array([-1, -1, -1]), high=np.array([1, 1, 1]), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)

        self.agent_position = np.array([1.0, 1.0], dtype=np.float32)
        self.agent_angle = 0.0
        self.path = []

        self.grid_width, self.grid_height = self._calculate_grid_size(bounding_box_points)
        self.visited_grid = np.zeros((self.grid_width, self.grid_height), dtype=bool)

        self.screen = None
        self.clock = pygame.time.Clock()
        self.field_color = (107, 142, 35)  # Olive Drab
        self.covered_color = (34, 139, 34)  # Forest Green
        self.outside_color = (139, 69, 19)  # Saddle Brown
        self.path_color = (255, 0, 0)  # Red
        self.field_border_color = (0, 0, 0)  # Black

        self.tractor_img = pygame.image.load('tractor.png')
        self.tractor_img = pygame.transform.scale(self.tractor_img, (self.cell_size * 30, self.cell_size * 30))

    def _calculate_grid_size(self, bounding_box_points):
        x_coords, y_coords = zip(*bounding_box_points)
        width = int((max(x_coords) - min(x_coords)) / self.cell_size) + 1
        height = int((max(y_coords) - min(y_coords)) / self.cell_size) + 1
        return width, height

    def _get_grid_cell(self, position):
        x, y = position
        cell_x = int(x / self.cell_size)
        cell_y = int(y / self.cell_size)
        return cell_x, cell_y

    def _clamp_position_to_bounds(self, position):
        x, y = position
        clamped_x = np.clip(x, 0, self.grid_width * self.cell_size - 1)
        clamped_y = np.clip(y, 0, self.grid_height * self.cell_size - 1)
        return np.array([clamped_x, clamped_y], dtype=np.float32)

    def _normalize_observation(self):
        normalized_position = self.agent_position / np.array(
            [self.grid_width * self.cell_size, self.grid_height * self.cell_size], dtype=np.float32)
        normalized_angle = self.agent_angle / (2 * np.pi)
        return np.concatenate([normalized_position, [normalized_angle]], dtype=np.float32)

    def step(self, action):
        self.current_step += 1

        throttle, steering, stop = action

        if stop >= 0:
            terminated = True
            truncated = self.current_step >= self.max_steps
            return self._normalize_observation(), 0, terminated, truncated, {}

        self.agent_angle += steering
        self.agent_angle %= 2 * np.pi

        new_position = self.agent_position + np.array(
            [np.cos(self.agent_angle) * throttle, np.sin(self.agent_angle) * throttle], dtype=np.float32)
        new_position = self._clamp_position_to_bounds(new_position)

        reward = 0

        if self._is_inside_polygon(new_position):
            self.agent_position = new_position
            reward = 1

            cell_x, cell_y = self._get_grid_cell(self.agent_position)
            self.visited_grid[cell_x, cell_y] = True
        else:
            self.agent_position = new_position
            reward = -1

        self.path.append(tuple(self.agent_position))

        coverage = self.calculate_coverage()
        if coverage >= self.coverage_threshold:
            reward += 10
            terminated = True
        else:
            terminated = False

        truncated = self.current_step >= self.max_steps

        return self._normalize_observation(), reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_position = np.array([1.0, 1.0], dtype=np.float32)
        self.agent_angle = 0.0
        self.current_step = 0
        self.visited_grid = np.zeros((self.grid_width, self.grid_height), dtype=bool)
        self.path = [tuple(self.agent_position)]
        return self._normalize_observation(), {}

    def _get_observation(self):
        return self._normalize_observation()

    def _is_inside_polygon(self, point):
        return self.field_polygon.contains(Point(point))

    def calculate_coverage(self):
        return np.sum(self.visited_grid) / self.visited_grid.size

    def render(self):
        if self.render_mode is None:
            return

        if self.screen is None:
            pygame.init()
            x_coords, y_coords = zip(*self.bounding_box.exterior.coords)
            width = int((max(x_coords) - min(x_coords)) * self.cell_size * 50)
            height = int((max(y_coords) - min(y_coords)) * self.cell_size * 50)
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption('Polygon Coverage Env')

        self.screen.fill(self.outside_color)

        scaled_field_points = [(int(x * self.cell_size * 50), int(y * self.cell_size * 50)) for x, y in
                               self.field_polygon.exterior.coords]
        pygame.draw.polygon(self.screen, self.field_color, scaled_field_points, 0)

        pygame.draw.polygon(self.screen, self.field_border_color, scaled_field_points, 1)

        if len(self.path) > 1:
            scaled_path_points = [(int(x * self.cell_size * 50), int(y * self.cell_size * 50)) for x, y in self.path]
            pygame.draw.lines(self.screen, self.path_color, False, scaled_path_points, 2)

        rotated_tractor = pygame.transform.rotate(self.tractor_img, -np.degrees(self.agent_angle))
        rect = rotated_tractor.get_rect(center=(int(self.agent_position[0] * self.cell_size * 50 + self.cell_size * 25),
                                                int(self.agent_position[
                                                        1] * self.cell_size * 50 + self.cell_size * 25)))
        self.screen.blit(rotated_tractor, rect.topleft)

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None


if __name__ == '__main__':
    field_points = [(1, 1), (6, 1), (8, 5), (5, 8), (1, 6)]
    bounding_box_points = [(0, 0), (10, 0), (10, 10), (0, 10)]

    env = ContinuousPolygonCoverageEnv(field_points=field_points, bounding_box_points=bounding_box_points,
                                       render_mode='human')
    env = Monitor(env)

    check_env(env, warn=True)

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_tensorboard/")

    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./logs/', name_prefix='ppo_model')
    total_timesteps = 50000
    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback])

    model.save("ppo_final_model")

    eval_env = ContinuousPolygonCoverageEnv(field_points=field_points, bounding_box_points=bounding_box_points,
                                            cell_size=1, render_mode='human')
    eval_env = Monitor(eval_env)
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, render=True)

    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    eval_env.close()