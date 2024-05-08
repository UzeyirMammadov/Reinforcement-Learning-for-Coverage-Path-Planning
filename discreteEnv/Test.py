import numpy as np

from discreteEnv.Agent import QLearningAgent, train_agent
from discreteEnv.DiscreteEnvironment import CoverageEnv


def test_agent(env, agent, episodes=10):
    total_rewards = 0
    for episode in range(episodes):
        env.reset()
        done = False
        while not done:
            state_index = np.ravel_multi_index(env.agent_position, env.grid_size)
            action = np.argmax(agent.q_table[state_index])
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_rewards += reward
        print(f"Test Episode {episode + 1}: Reward {reward}")
    average_reward = total_rewards / episodes
    print(f"Average Reward: {average_reward}")
    return average_reward


env = CoverageEnv()
agent = QLearningAgent(action_space=5, state_space=np.prod(env.grid_size))

print("Testing before training:")
test_agent(env, agent, episodes=10)

print("Training the agent:")
train_agent(1000, env)

print("Testing after training:")
test_agent(env, agent, episodes=10)

