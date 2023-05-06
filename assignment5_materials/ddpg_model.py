import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce

class actor(nn.Module):
    def __init__(self, action_size) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.bn4 = nn.BatchNorm2d(128)
        self.head = nn.Linear(128, action_size)
    
    def forward(self, x):
        
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = F.tanh(self.bn4(self.conv4(x))) # [-1, 1]
        x = reduce(x, 'b c h w -> b c', reduction='mean')
        
        return self.head(x)
    
class critic(nn.Module):
    def __init__(self, action_size, hidde_size=64, hidde_size1=64) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128, hidde_size)

        self.fc2 = nn.Linear(hidde_size + action_size, hidde_size1)
        self.fc3 = nn.Linear(hidde_size1, 1)

    
    def forward(self, state, action):
        
        state = F.leaky_relu(self.bn1(self.conv1(state)))
        state = F.leaky_relu(self.bn2(self.conv2(state)))
        state = F.leaky_relu(self.bn3(self.conv3(state)))
        state = reduce(state, 'b c h w -> b c', reduction='mean')
        state = F.leaky_relu(self.fc1(state))

        out = F.leaky_relu(self.fc2(torch.cat([state, action], dim=1)))
        out = F.leaky_relu(self.fc3(out))

        return out