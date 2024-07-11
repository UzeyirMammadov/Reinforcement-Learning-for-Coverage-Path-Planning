import gymnasium
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.noise import NormalActionNoise
from torch.utils.tensorboard import SummaryWriter

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
    'learning_rate': 0.001,
    'buffer_size': 25000,
    'exploration_initial_eps': 1.0,
    'exploration_fraction': 0.2,
    'exploration_final_eps': 0.2,
    'target_update_interval': 5000,
    'batch_size' : 64

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
    def __init__(self, grid_size=(5, 5), cell_size=50):
        super(CoverageEnv, self).__init__()
        self.action_space = gymnasium.spaces.Discrete(4)
        self.observation_space = gymnasium.spaces.Box(low=0, high=1, shape=(grid_size[0] * grid_size[1],), dtype=np.float32)
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.grid = np.zeros(grid_size, dtype=np.float32)
        self.agent_position = [0, 0]
        self.max_steps = 500
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
        reward = -0.1

        if self.grid[new_position] == 0:
            reward += 1.5
            self.grid[new_position] = 1
        elif self.grid[new_position] == 1:
            reward -= 0.1

        # coverage_percentage = np.sum(self.grid) / (self.grid_size[0] * self.grid_size[1])

        # if coverage_percentage >= 0.25 and coverage_percentage < 0.30:
        #     reward += 0.5 
        # elif coverage_percentage >= 0.50 and coverage_percentage < 0.55:
        #     reward += 0.7 
        # elif coverage_percentage >= 0.90 and coverage_percentage < 1.0:
        #     reward += 1

        if np.all(self.grid == 1):
            reward += 10
            # if self.last_episode_steps is None or self.current_step < self.last_episode_steps:
            #     reward += 10 if self.last_episode_steps is not None else 0
            #     self.last_episode_steps = self.current_step
            terminated = True
        else:
            terminated = False

        truncated = self.current_step >= self.max_steps

        # print(self.agent_position)
        # print(self.grid)

        print(f"Step: {self.current_step}, Action: {action}, Position: {self.agent_position}, Reward: {reward}")


        flattened_grid = self.grid.flatten()

        return flattened_grid, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = np.zeros(self.grid_size, dtype=np.float32)
        self.agent_position = [0, 0]
        self.current_step = 0
        return self.grid.flatten(), {}

env = CoverageEnv(grid_size=(5, 5))
check_env(env, warn=True)

env = Monitor(env)
env = DummyVecEnv([lambda: env])

model = DQN('MlpPolicy', env, verbose=1, tensorboard_log="./dqn_tensorboard/", 
        learning_rate=hyperparams['learning_rate'],
        buffer_size=hyperparams['buffer_size'],
        exploration_initial_eps=hyperparams['exploration_initial_eps'],
        exploration_fraction=hyperparams['exploration_fraction'],
        exploration_final_eps=hyperparams['exploration_final_eps'],
        target_update_interval=hyperparams['target_update_interval'],
        batch_size=hyperparams['batch_size'])

checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./logs/', name_prefix='dqn_model')
reward_logger_callback = RewardLoggerCallback()
hyperparameter_logger_callback = HyperparameterLoggerCallback(hyperparams)

model.learn(total_timesteps=1000000, callback=[checkpoint_callback, reward_logger_callback, hyperparameter_logger_callback])

model.save("dqn_final_model")

env.close()

eval_env = CoverageEnv(grid_size=(5, 5))
eval_env = DummyVecEnv([lambda: Monitor(eval_env)])

eval_callback = EvalCallback(eval_env, eval_freq=10000)

model = DQN.load("dqn_final_model")

eval_writer = SummaryWriter(log_dir="./dqn_tensorboard/evaluation")

obs = eval_env.reset()
total_steps = 0 
max_total_steps = 1000  
episode_rewards = []
current_episode_reward = 0

for step in range(max_total_steps):
    action, _states = model.predict(obs, deterministic=True)
    result = eval_env.step(action)
    if len(result) == 5:
        obs, rewards, terminated, truncated, info = result
    else:
        obs, rewards, done, info = result
        terminated = done
        truncated = False

    total_steps += 1
    current_episode_reward += rewards
    
    if terminated or truncated:
        print(f"Episode completed in {total_steps} steps with reward: {current_episode_reward}")
        episode_rewards.append(current_episode_reward)
        current_episode_reward = 0
        obs = eval_env.reset()

mean_eval_reward = np.mean(episode_rewards)
eval_writer.add_scalar('evaluation/mean_reward', mean_eval_reward, total_steps)
eval_writer.close()

eval_env.close()