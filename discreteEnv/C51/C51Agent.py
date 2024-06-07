import torch
import torch.optim as optim
import numpy as np
import random
from collections import deque

from discreteEnv.C51.C51Network import C51Network

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class C51Agent:
    def __init__(self, state_dim, action_dim, num_atoms=51, v_min=-10, v_max=10, lr=1e-3):
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (num_atoms - 1)
        self.z = torch.linspace(v_min, v_max, num_atoms)

        self.model = C51Network(state_dim, action_dim, num_atoms, v_min, v_max).to(device)
        self.target_model = C51Network(state_dim, action_dim, num_atoms, v_min, v_max).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.replay_buffer = deque(maxlen=100000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01
        self.update_target_every = 1000
        self.steps = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = (self.model(state) * self.z).sum(dim=2)
        return q_values.argmax().item()

    def add_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        state, action, reward, next_state, done = zip(*batch)

        state = torch.FloatTensor(np.array(state)).to(device)
        action = torch.LongTensor(np.array(action)).unsqueeze(1).to(device)
        reward = torch.FloatTensor(np.array(reward)).unsqueeze(1).to(device)
        next_state = torch.FloatTensor(np.array(next_state)).to(device)
        done = torch.FloatTensor(np.array(done)).unsqueeze(1).to(device)

        batch_indices = np.arange(self.batch_size, dtype=np.int64)

        next_dist = self.target_model(next_state).detach()
        next_action = (next_dist * self.z).sum(dim=2).argmax(dim=1)
        next_dist = next_dist[batch_indices, next_action]

        target_z = reward + self.gamma * self.z * (1 - done)
        target_z = target_z.clamp(min=self.v_min, max=self.v_max)

        b = (target_z - self.v_min) / self.delta_z
        l = b.floor().long()
        u = b.ceil().long()

        m = torch.zeros(self.batch_size, self.num_atoms).to(device)
        offset = torch.linspace(0, (self.batch_size - 1) * self.num_atoms, self.batch_size).long().unsqueeze(1).expand(
            self.batch_size, self.num_atoms).to(device)

        m.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
        m.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))

        dist = self.model(state)
        log_p = torch.log(dist[batch_indices, action.view(-1)])
        loss = -(m * log_p).sum(dim=1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps % self.update_target_every == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        self.steps += 1