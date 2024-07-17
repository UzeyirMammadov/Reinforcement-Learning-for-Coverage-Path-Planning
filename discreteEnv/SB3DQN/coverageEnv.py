import gymnasium
import numpy as np
import pygame
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy


class EvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq, n_eval_episodes=5, verbose=0):
        super(EvalCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            episode_rewards = []
            for _ in range(self.n_eval_episodes):
                obs = self.eval_env.reset()
                current_episode_reward = 0
                done = False
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                    current_episode_reward += reward
                    done = terminated or truncated
                episode_rewards.append(current_episode_reward)
            mean_reward = np.mean(episode_rewards)
            self.logger.record('evaluation/mean_reward', mean_reward, self.num_timesteps)
            self.logger.dump(self.num_timesteps)
        return True


class HyperparameterLoggerCallback(BaseCallback):
    def __init__(self, hyperparams, verbose=0):
        super(HyperparameterLoggerCallback, self).__init__(verbose)
        self.hyperparams = hyperparams

    def _on_training_start(self) -> None:
        for key, value in self.hyperparams.items():
            self.logger.record(f'hyperparams/{key}', value)

    def _on_step(self) -> bool:
        return True


hyperparams = {
    'learning_rate': 0.0002,
    'buffer_size': 3000,
    'target_update_interval': 5000,
    'exploration_fraction': 0.3
}


class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLoggerCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.episode_count = 0

    def _on_step(self) -> bool:
        self.current_episode_reward += self.locals['rewards'][0]
        
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.logger.record('rollout/ep_rew_mean', np.mean(self.episode_rewards))
            self.logger.record('episode/reward', self.current_episode_reward)

            self.current_episode_reward = 0
            self.episode_count += 1

        return True


class CoverageEnv(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, grid_size=(5, 5), cell_size=50, render_mode=None):
        super(CoverageEnv, self).__init__()
        self.action_space = gymnasium.spaces.Discrete(5)
        self.observation_space = gymnasium.spaces.Box(low=0, high=1, shape=(grid_size[0] * grid_size[1],),
                                                      dtype=np.float32)
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.grid = np.zeros(grid_size, dtype=np.float32)
        self.agent_position = [0, 0]
        self.max_steps = 100
        self.current_step = 0
        self.last_episode_steps = None
        self.window = None
        self.clock = None
        self.render_mode = render_mode

        self._action_to_direction = {
            0: np.array([1, 0]),  # right
            1: np.array([0, 1]),  # up
            2: np.array([-1, 0]),  # left
            3: np.array([0, -1]),  # down
            4: "Finished"  # stop
        }

    def step(self, action):
        self.current_step += 1
        prev_position = tuple(self.agent_position.copy())

        if action == 4: 
            terminated = True
            truncated = self.current_step >= self.max_steps
            return self._get_obs(), 0, terminated, truncated, {}

        direction = self._action_to_direction[action]
        self.agent_position = np.clip(self.agent_position + direction, 0, self.grid_size[0] - 1)

        new_position = tuple(self.agent_position)

        reward = 0
        if new_position == prev_position:
            reward = -1  
        elif self.grid[new_position] == 0:
            reward = 10 
            self.grid[new_position] = 1
        else:
            reward = -1 

        if np.all(self.grid == 1):
            reward += 20
            terminated = True
        else:
            terminated = False

        truncated = self.current_step >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = np.zeros(self.grid_size, dtype=np.float32)
        self.agent_position = np.array([0, 0])
        self.current_step = 0
        return self._get_obs(), {}

    def _get_obs(self):
        return self.grid.flatten()

    def render(self, mode='human'):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode(
                (self.grid_size[1] * self.cell_size, self.grid_size[0] * self.cell_size))
            self.clock = pygame.time.Clock()

        self.window.fill((255, 255, 255))

        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                color = (0, 0, 0) if self.grid[i, j] == 1 else (255, 255, 255)
                pygame.draw.rect(self.window, color,
                                 (j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size))
                pygame.draw.rect(self.window, (200, 200, 200),
                                 (j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size), 1)

        pygame.draw.circle(self.window, (0, 0, 255), (int(self.agent_position[1] * self.cell_size + self.cell_size / 2),
                                                      int(self.agent_position[
                                                              0] * self.cell_size + self.cell_size / 2)),
                           self.cell_size // 3)

        pygame.display.flip()
        self.clock.tick(5)

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None
            self.clock = None


env = CoverageEnv(grid_size=(5, 5))
check_env(env, warn=True)

env = Monitor(env)
env = DummyVecEnv([lambda: env])

model = DQN('MlpPolicy', env, verbose=1, tensorboard_log="./dqn_tensorboard/",
            learning_rate=hyperparams['learning_rate'],
            buffer_size=hyperparams['buffer_size'],
            target_update_interval=hyperparams['target_update_interval'],
            exploration_fraction=hyperparams['exploration_fraction'])

checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./logs/', name_prefix='dqn_model')
reward_logger_callback = RewardLoggerCallback()
hyperparameter_logger_callback = HyperparameterLoggerCallback(hyperparams)

model.learn(total_timesteps=1000000,
            callback=[checkpoint_callback, reward_logger_callback, hyperparameter_logger_callback])

model.save("dqn_final_model")

env.close()

eval_env = CoverageEnv(grid_size=(5, 5), render_mode='human')
eval_env = DummyVecEnv([lambda: Monitor(eval_env)])

mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, render=True)
print(f"Mean reward: {mean_reward} +/- {std_reward:.2f}")

eval_env.close()