import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from shapely.geometry import Point, Polygon, box
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy

class ContinuousPolygonCoverageEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self, field_points, bounding_box_points, grid_resolution=10, max_steps=100, coverage_threshold=1,
                 render_mode=None, tractor_width=None, scale_factor=None):
        super(ContinuousPolygonCoverageEnv, self).__init__()
        self.field_polygon = Polygon(field_points)
        self.bounding_box = Polygon(bounding_box_points)
        self.grid_resolution = grid_resolution
        self.cell_size = tractor_width
        self.max_steps = max_steps
        self.current_step = 0
        self.coverage_threshold = coverage_threshold
        self.render_mode = render_mode
        self.reward = 0
        self.tractor_width = tractor_width
        self.scale_factor = scale_factor

        self.action_space = spaces.Box(low=np.array([-1, -1, -1]), high=np.array([1, 1, 1]), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.grid_resolution * self.grid_resolution + 4,), dtype=np.float32)

        self.agent_position = np.array([0.5, 0.5], dtype=np.float32)  # Adjusted for smaller environment
        self.agent_angle = 0.0
        self.path = []

        self.grid_width, self.grid_height = self.grid_resolution, self.grid_resolution

        # Only create grid cells within the field polygon
        self.visited_grid = np.zeros((self.grid_width, self.grid_height), dtype=bool)
        self.valid_grid_cells = self._create_valid_grid_cells()
        self.overlap_count = 0

        self.screen = None
        self.clock = pygame.time.Clock()
        self.field_color = (107, 142, 35)  # Olive Drab
        self.covered_color = (34, 139, 34)  # Forest Green
        self.outside_color = (139, 69, 19)  # Saddle Brown
        self.visited_color = (255, 0, 0)  # Red
        self.field_border_color = (0, 0, 0)  # Black
        self.tractor_color = (0, 0, 255)  # Blue
        self.tractor_border_color = (0, 0, 0)  # Black

        self.coverage_20_flag = False
        self.coverage_40_flag = False
        self.coverage_60_flag = False
        self.coverage_80_flag = False

    def _create_valid_grid_cells(self):
        valid_cells = []
        minx, miny, maxx, maxy = self.field_polygon.bounds
        x = minx
        while x + self.cell_size <= maxx:
            y = miny
            while y + self.cell_size <= maxy:
                cell = box(x, y, x + self.cell_size, y + self.cell_size)
                if self.field_polygon.contains(cell):
                    cell_x = int((x - minx) / self.cell_size)
                    cell_y = int((y - miny) / self.cell_size)
                    valid_cells.append((cell_x, cell_y))
                y += self.cell_size
            x += self.cell_size
        return valid_cells

    def _get_grid_cells(self, position, width):
        x, y = position
        cells = []
        half_width = width / 2
        for dx in np.arange(-half_width, half_width, self.cell_size):
            for dy in np.arange(-half_width, half_width, self.cell_size):
                cell_x = int((x + dx) // self.cell_size)
                cell_y = int((y + dy) // self.cell_size)
                if (cell_x, cell_y) in self.valid_grid_cells:  # Only add valid grid cells
                    cells.append((cell_x, cell_y))
        return cells

    def _normalize_observation(self):
        normalized_position = self.agent_position / np.array(
            [self.grid_width * self.cell_size, self.grid_height * self.cell_size], dtype=np.float32)
        normalized_angle = self.agent_angle / (2 * np.pi)
        flattened_grid = self.visited_grid.flatten().astype(np.float32)
        nearest_unvisited = self._nearest_unvisited_distance() / np.sqrt(self.grid_width**2 + self.grid_height**2)
        return np.concatenate([flattened_grid, normalized_position, [normalized_angle, nearest_unvisited]], dtype=np.float32)

    def _nearest_unvisited_distance(self):
        distances = []
        for cell_x, cell_y in self.valid_grid_cells:
            cell_center = np.array([cell_x * self.cell_size + self.cell_size / 2, cell_y * self.cell_size + self.cell_size / 2])
            if not self.visited_grid[cell_x, cell_y]:
                distances.append(np.linalg.norm(self.agent_position - cell_center))
        return min(distances) if distances else 0

    def step(self, action):
        self.current_step += 1

        throttle, steering, stop = action

        if stop >= 0:
            terminated = True
            truncated = self.current_step >= self.max_steps
            return self._normalize_observation(), 0, terminated, truncated, {}

        self.agent_angle += steering
        self.agent_angle %= 2 * np.pi

        if steering != 0:
            throttle *= 0.3

        new_position = self.agent_position + np.array(
            [np.cos(self.agent_angle) * throttle, np.sin(self.agent_angle) * throttle], dtype=np.float32)

        coverage = self.calculate_coverage()

        if self._is_inside_polygon(new_position):
            self.agent_position = new_position

            cells = self._get_grid_cells(self.agent_position, self.tractor_width)
            for cell_x, cell_y in cells:
                if 0 <= cell_x < self.grid_width and 0 <= cell_y < self.grid_height:
                    if not self.visited_grid[cell_x, cell_y]:
                        self.reward = 5 + (5 * coverage >= 0.80)
                        self.visited_grid[cell_x, cell_y] = True
                    else:
                        self.reward = -1
                        self.overlap_count += 1
        else:
            self.agent_position = new_position
            self.reward = -1
            terminated = True
            truncated = self.current_step >= self.max_steps
            return self._normalize_observation(), self.reward, terminated, truncated, {}

        self.path.append(tuple(self.agent_position))

        if coverage >= 0.8:
            print("80%")

        if coverage >= self.coverage_threshold:
            self.reward += 10
            print("Threshold reached")
            terminated = True
            truncated = self.current_step >= self.max_steps
            return self._normalize_observation(), self.reward, terminated, truncated, {}
        else:
            terminated = False

        truncated = self.current_step >= self.max_steps

        return self._normalize_observation(), self.reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_position = np.array([0.5, 0.5], dtype=np.float32)
        self.agent_angle = 0.0
        self.current_step = 0
        self.visited_grid = np.zeros((self.grid_width, self.grid_height), dtype=bool)
        self.path = [tuple(self.agent_position)]
        self.overlap_count = 0

        self.coverage_20_flag = False
        self.coverage_40_flag = False
        self.coverage_60_flag = False
        self.coverage_80_flag = False

        return self._normalize_observation(), {}

    def _is_inside_polygon(self, point):
        return self.field_polygon.contains(Point(point))

    def calculate_coverage(self):
        visited_cells_in_polygon = sum(self.visited_grid[cell_x, cell_y] for cell_x, cell_y in self.valid_grid_cells)
        return visited_cells_in_polygon / len(self.valid_grid_cells)

    def render(self):
        if self.render_mode is None:
            return

        if self.screen is None:
            pygame.init()
            x_coords, y_coords = zip(*self.bounding_box.exterior.coords)
            width = int((max(x_coords) - min(x_coords)) * self.scale_factor)
            height = int((max(y_coords) - min(y_coords)) * self.scale_factor)
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption('Polygon Coverage Env')

        self.screen.fill(self.outside_color)

        scaled_field_points = [(int(x * self.scale_factor), int(y * self.scale_factor)) for x, y in
                               self.field_polygon.exterior.coords]
        pygame.draw.polygon(self.screen, self.field_color, scaled_field_points, 0)

        for i in range(self.grid_width):
            for j in range(self.grid_height):
                cell_x = i * self.cell_size
                cell_y = j * self.cell_size
                if self.visited_grid[i, j]:
                    pygame.draw.rect(self.screen, self.visited_color,
                                     (int(cell_x * self.scale_factor), int(cell_y * self.scale_factor),
                                      int(self.cell_size * self.scale_factor), int(self.cell_size * self.scale_factor)))

        pygame.draw.polygon(self.screen, self.field_border_color, scaled_field_points, 1)

        if len(self.path) > 1:
            scaled_path_points = [(int(x * self.scale_factor), int(y * self.scale_factor)) for x, y in self.path]
            pygame.draw.lines(self.screen, (0, 0, 0), False, scaled_path_points, 2)

        tractor_width = self.tractor_width * self.scale_factor
        tractor_length = self.cell_size * self.scale_factor * 1.5
        center_x = int(self.agent_position[0] * self.scale_factor)
        center_y = int(self.agent_position[1] * self.scale_factor)

        tractor_surface = pygame.Surface((tractor_length, tractor_width), pygame.SRCALPHA)
        tractor_surface.fill(self.tractor_color)
        pygame.draw.rect(tractor_surface, self.tractor_border_color, tractor_surface.get_rect(), 1)

        rotated_surface = pygame.transform.rotate(tractor_surface, -np.degrees(self.agent_angle))

        rect = rotated_surface.get_rect(center=(center_x, center_y))

        self.screen.blit(rotated_surface, rect.topleft)

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
    scale_factor = 0.5
    tractor_width= 0.5
    field_points = [(1 * scale_factor, 1 * scale_factor), 
                    (6 * scale_factor, 1 * scale_factor), 
                    (8 * scale_factor, 5 * scale_factor), 
                    (5 * scale_factor, 8 * scale_factor), 
                    (1 * scale_factor, 6 * scale_factor)]
    bounding_box_points = [(0 * scale_factor, 0 * scale_factor), 
                        (10 * scale_factor, 0 * scale_factor), 
                        (10 * scale_factor, 10 * scale_factor), 
                        (0 * scale_factor, 10 * scale_factor)]

    grid_resolution = 10
    scale_factor_render = 50
    env = ContinuousPolygonCoverageEnv(
        field_points=field_points, 
        bounding_box_points=bounding_box_points,
        grid_resolution=grid_resolution,
        scale_factor=scale_factor_render,
        tractor_width=tractor_width)
    env = Monitor(env)
    check_env(env, warn=True)

    model = PPO("MlpPolicy", 
                env, 
                verbose=1, 
                tensorboard_log="./continuousEnv/PPO/ppo_tensorboard/", 
                learning_rate=0.0003,
                clip_range=0.1,
                vf_coef=0.3,
                max_grad_norm=0.5,
                normalize_advantage=True)

    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./continuousEnv/PPO/logs/', name_prefix='ppo_model')
    total_timesteps = 3000000
    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback])

    model.save("continuousEnv/PPO/ppo_final_model")

    env.close()

    eval_env = ContinuousPolygonCoverageEnv(field_points=field_points, 
                                            bounding_box_points=bounding_box_points,
                                            grid_resolution=grid_resolution,
                                            scale_factor=scale_factor_render,
                                            tractor_width=tractor_width)
    eval_env = Monitor(eval_env)
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, render=False, deterministic=True)

    total_coverage = []
    total_overlap = []

    for _ in range(10):
        obs, _ = eval_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
        total_coverage.append(eval_env.calculate_coverage())
        total_overlap.append(eval_env.overlap_count / eval_env.current_step)

    avg_coverage = np.mean(total_coverage)
    avg_overlap = np.mean(total_overlap)

    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Average Coverage: {avg_coverage * 100:.2f}%")
    print(f"Average Overlap: {avg_overlap * 100:.2f}%")

    eval_env.close()
