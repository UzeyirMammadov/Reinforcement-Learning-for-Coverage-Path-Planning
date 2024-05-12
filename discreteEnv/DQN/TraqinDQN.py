import os
import time

import torch

from discreteEnv.DQN.CoverageEnv import CoverageEnv
from discreteEnv.DQN.DQNAgent import DQNAgent
from discreteEnv.DQN.ReplayBuffer import ReplayBuffer

env = CoverageEnv(grid_size=(5, 5))
state_shape = (1,) + env.observation_space.shape
agent = DQNAgent(state_shape=state_shape, action_space=env.action_space.n)
replay_buffer = ReplayBuffer(20000)
def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)

def load_checkpoint(filename="checkpoint.pth.tar"):
    return torch.load(filename)

def train_dqn(episodes, start_episode=0):
    global_step = 0
    for episode in range(start_episode, episodes):
        state = env.reset().reshape(state_shape)
        done = False
        total_reward = 0
        steps = 0
        while not done:
            env.render()
            time.sleep(0.1)
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.reshape(state_shape)
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps += 1
            if len(replay_buffer.buffer) > 500:
                batch = replay_buffer.sample(32)
                for s, a, r, s_next, d in batch:
                    agent.train(s, a, r, s_next, d)
                    loss = agent.loss_fn(
                        agent.model(torch.tensor(s, dtype=torch.float32).to(agent.device).unsqueeze(0)),
                        agent.model(torch.tensor(s, dtype=torch.float32).to(agent.device).unsqueeze(0)))
                    agent.writer.add_scalar('Loss', loss.item(), global_step)
                    global_step += 1
            if done:
                print(f"Episode {episode + 1}: Total reward = {total_reward}, Epsilon = {agent.epsilon:.2f}")

            agent.writer.add_scalar('Epsilon', agent.epsilon, episode)
            agent.writer.add_scalar('Total Reward', total_reward, episode)
            agent.writer.add_scalar('Buffer Size', len(replay_buffer.buffer), episode)

        # Save checkpoint
        checkpoint = {
            'episode': episode,
            'model_state_dict': agent.model.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'epsilon': agent.epsilon
        }
        save_checkpoint(checkpoint)

    env.close()

# Load checkpoint if exists
start_episode = 0
if os.path.exists("checkpoint.pth.tar"):
    checkpoint = load_checkpoint()
    agent.model.load_state_dict(checkpoint['model_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_episode = checkpoint['episode'] + 1
    agent.epsilon = checkpoint['epsilon']

train_dqn(20, start_episode)
