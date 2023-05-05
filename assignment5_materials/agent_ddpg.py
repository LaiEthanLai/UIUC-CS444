import random
import torch
import numpy as np
from collections import deque
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from memory import ReplayMemory, ReplayMemoryLSTM
from model import actor, critic
from utils import find_max_lives, check_live, get_frame, get_init_state
from config import *
from OrnsteinUhlenbeckProcess import OrnsteinUhlenbeckProcess
device = 'cpu'
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = 'mps'
if torch.cuda.is_available():
    device = 'cuda'

class Agent():
    def __init__(self, action_size):

        print(f'training on {device}')

        self.action_size = action_size

        # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.explore_step = 500000
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.explore_step
        self.train_start = 100000
        self.update_target = 1000
        self.tau = 0.005

        # Generate the memory
        self.memory = ReplayMemory()

        # Create actor, target actor
        self.actor = actor(action_size)
        self.actor.to(device)
        self.target_actor = actor(action_size)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_actor.to(device)
        self.target_actor.eval()

        # Create critic, target critic
        self.critic = critic(action_size)
        self.critic.to(device)
        self.target_critic = critic(action_size)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_critic.to(device)
        self.target_critic.eval()

        self.actor_optim = optim.Adam(params=self.actor.parameters(), lr=learning_rate)
        self.critic_optim = optim.Adam(params=self.critic.parameters(), lr=learning_rate)
        self.actor_scheduler = optim.lr_scheduler.StepLR(self.actor_optim, step_size=scheduler_step_size, gamma=scheduler_gamma)
        self.critic_scheduler = optim.lr_scheduler.StepLR(self.critic_optim, step_size=scheduler_step_size, gamma=scheduler_gamma)

        self.random_process = OrnsteinUhlenbeckProcess(action_size=action_size)

    
    def soft_update(self, target_net, net):

        for param_target, param_policy in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target + self.tau * (param_policy - param_target))

    def load_policy_net(self, path):
        self.actor = torch.load(path)

    """Get action using policy net using epsilon-greedy policy"""
    def get_action(self, state):
        state = torch.from_numpy(state).to(device)[None, :]
        self.actor.eval()
        with torch.no_grad():
            act_score = self.actor(state).cpu()
            act_score += max(self.epsilon, 0) * self.random_process.sample()
        self.actor.train()
        self.epsilon -= self.epsilon_decay
        return act_score

    # pick samples randomly from replay memory (with batch_size)
    def train_policy_net(self, frame):

        mini_batch = self.memory.sample_mini_batch(frame)
        mini_batch = np.array(mini_batch, object).transpose()

        history = np.stack(mini_batch[0], axis=0)
        states = np.float32(history[:, :4, :, :]) / 255.
        states = torch.from_numpy(states).to(device)
        
        actions = list(mini_batch[1])
        actions = torch.cat(actions).to(device)
        rewards = list(mini_batch[2])
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.from_numpy(np.float32(history[:, 1:, :, :]))/ 255.
        next_states = next_states.to(device)
        dones = mini_batch[3] # checks if the game is over
        mask = torch.tensor(list(map(int, dones==False)),dtype=torch.uint8).to(device)
        
        target= self.target_critic(next_states, self.target_actor(states).detach()).squeeze() * self.discount_factor * mask + rewards
        criterion = nn.SmoothL1Loss()

        # train critic
        self.critic_optim.zero_grad()   
        loss_critic = criterion(target, self.critic(states, actions).squeeze())
        loss_critic.backward()
        self.critic_optim.step()
        self.critic_scheduler.step()

        # train actor
        self.actor_optim.zero_grad()

        loss_actor = -1 * self.critic(states, self.actor(states).squeeze()).mean()
        loss_actor.backward()
        self.actor_optim.step()
        self.actor_scheduler.step()

        # update target
        self.soft_update(target_net=self.target_actor, net=self.actor)
        self.soft_update(target_net=self.target_critic, net=self.critic)
