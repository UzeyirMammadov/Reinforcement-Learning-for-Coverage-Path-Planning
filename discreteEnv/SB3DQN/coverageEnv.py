import gymnasium
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

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
    def __init__(self, grid_size=(5, 5), cell_size=50):
        super(CoverageEnv, self).__init__()
        self.action_space = gymnasium.spaces.Discrete(4)
        self.observation_space = gymnasium.spaces.Box(low=0, high=1, shape=(grid_size[0] * grid_size[1],), dtype=np.int32)
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.grid = np.zeros(grid_size, dtype=np.int32)
        self.agent_position = [0, 0]
        self.max_steps = 2000
        self.current_step = 0
        self.last_episode_steps = None

    def step(self, action):
        self.current_step += 1
        prev_position = tuple(self.agent_position.copy())

        if action == 0:
            self.agent_position[0] = max(self.agent_position[0] - 1, 0)
        elif action == 1:
            self.agent_position[0] = min(self.agent_position[0] + 1, self.grid_size[0] - 1)
        elif action == 2:
            self.agent_position[1] = max(self.agent_position[1] - 1, 0)
        elif action == 3:
            self.agent_position[1] = min(self.agent_position[1] + 1, self.grid_size[1] - 1)

        new_position = tuple(self.agent_position)
        reward = 0

        if self.grid[new_position] == 0:
            reward = 1
        elif self.grid[new_position] == 1:
            reward = -1

        self.grid[new_position] = 1

        if np.all(self.grid == 1):
            reward += 2
            if self.last_episode_steps is None or self.current_step < self.last_episode_steps:
                reward += 2 if self.last_episode_steps is not None else 0
                self.last_episode_steps = self.current_step
            terminated = True
        else:
            terminated = False

        truncated = self.current_step >= self.max_steps

        if new_position == prev_position:
            reward -= 0.5

        flattened_grid = self.grid.flatten()


        return flattened_grid, reward, terminated, truncated, {}

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.grid = np.zeros(self.grid_size, dtype=np.int32)
        self.agent_position = [0, 0]
        self.current_step = 0
        return self.grid.flatten(), {}

env = CoverageEnv(grid_size=(5, 5))
check_env(env, warn=True)

env = Monitor(env)
env = DummyVecEnv([lambda: env])

model = DQN('MlpPolicy', env, verbose=1, tensorboard_log="./dqn_tensorboard/")
checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./logs/', name_prefix='dqn_model')
reward_logger_callback = RewardLoggerCallback()

model.learn(total_timesteps=500000, callback=[checkpoint_callback, reward_logger_callback])

model.save("dqn_final_model")
model = DQN.load("dqn_final_model")

obs = env.reset()
for i in range(1000):
    action = model.predict(obs, deterministic=True)
    obs, rewards, done, info = env.step(action)
    if done:
        obs = env.reset()

env.close()