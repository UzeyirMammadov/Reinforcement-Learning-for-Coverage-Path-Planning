import numpy as np
import random

class QLearningAgent:
    def __init__(self, action_space, state_space, learning_rate=0.1, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.99, min_exploration_rate=0.01):
        self.q_table = np.zeros((state_space, action_space))
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.min_epsilon = min_exploration_rate
        self.action_space = action_space

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_space - 1)
        else:
            return np.argmax(self.q_table[state])

    def update_q_value(self, state, action, reward, next_state):
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[state, action] = new_value

    def update_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

def train_agent(episodes, env):
    agent = QLearningAgent(action_space=5, state_space=np.prod(env.grid_size))
    for episode in range(episodes):
        state = env.reset().flatten()
        state_index = state.dot(1 << np.arange(state.size)[::-1])
        done = False
        while not done:
            action = agent.choose_action(state_index)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.flatten()
            next_state_index = next_state.dot(1 << np.arange(next_state.size)[::-1])
            agent.update_q_value(state_index, action, reward, next_state_index)
            state = next_state
            state_index = next_state_index
            agent.update_epsilon()
        print(f"Episode {episode + 1}: Epsilon {agent.epsilon:.2f}")
