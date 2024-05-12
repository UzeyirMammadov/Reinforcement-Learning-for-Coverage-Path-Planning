import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, state_shape, action_space):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(state_shape[0], 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * state_shape[1] * state_shape[2], 24)
        self.fc2 = nn.Linear(24, action_space)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)