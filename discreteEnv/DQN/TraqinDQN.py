import torch

from discreteEnv.DQN.CoverageEnv import CoverageEnv
from discreteEnv.DQN.DQNAgent import DQNAgent
from discreteEnv.DQN.ReplayBuffer import ReplayBuffer

env = CoverageEnv(grid_size=(5, 5))
state_shape = (1,) + env.observation_space.shape
agent = DQNAgent(state_shape=state_shape, action_space=env.action_space.n)
replay_buffer = ReplayBuffer(20000)

def train_dqn(episodes):
    for episode in range(episodes):
        state = env.reset().reshape(state_shape)
        done = False
        total_reward = 0
        while not done:
            env.render()
            time.sleep(0.1)
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.reshape(state_shape)
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if len(replay_buffer.buffer) > 500:
                batch = replay_buffer.sample(32)
                for s, a, r, s_next, d in batch:
                    agent.train(s, a, r, s_next, d)
                    loss = agent.loss_fn(
                        agent.model(torch.tensor(s, dtype=torch.float32).to(agent.device).unsqueeze(0)),
                        agent.model(torch.tensor(s, dtype=torch.float32).to(agent.device).unsqueeze(0)))
                    agent.writer.add_scalar('Loss', loss.item())
            if done:
                print(f"Episode {episode + 1}: Total reward = {total_reward}, Epsilon = {agent.epsilon:.2f}")

            agent.writer.add_scalar('Epsilon', agent.epsilon, episode)
            agent.writer.add_scalar('Total Reward', total_reward, episode)
            agent.writer.add_scalar('Buffer Size', len(replay_buffer.buffer), episode)

    env.close()

train_dqn(20)
