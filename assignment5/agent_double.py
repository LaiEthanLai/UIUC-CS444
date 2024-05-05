import random
import torch
import numpy as np
from collections import deque
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from memory import ReplayMemory
from model import DQN
from utils import find_max_lives, check_live, get_frame, get_init_state
from config import *
import os
from copy import deepcopy

device = 'cpu'
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = 'mps'
if torch.cuda.is_available():
    device = 'cuda'


class Agent():
    def __init__(self, action_size):
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

        # Create the policy net and the target net
        self.policy_net = DQN(action_size)
        self.policy_net.to(device)
        
        self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

        # Initialize a target network and initialize the target network to the policy net
        ### CODE ###
        self.target_net = DQN(action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.target_net.to(device)
        # self.freeze_net(self.target_net)

    def freeze_net(self, net: nn.Module):
        for param in net.parameters(): 
            param.requires_grad = False

    def load_policy_net(self, path):
        self.policy_net = torch.load(path)           

    # after some time interval update the target net to be same with policy net
    def update_target_net(self):
        ### CODE ###
        for param_target, param_policy in zip(self.target_net.parameters(), self.policy_net.parameters()):
            param_target.data.copy_(param_target + self.tau * (param_policy - param_target))

    """Get action using policy net using epsilon-greedy policy"""
    def get_action(self, state):
        self.policy_net.eval()
        if np.random.rand() <= self.epsilon:
            ### CODE #### 
            # Choose a random action
            act = random.choice([0, 1, 2])
        else:
            ### CODE ####
            # Choose the best action
            state = torch.from_numpy(state).to(device)[None, :]
            with torch.no_grad():
                act = self.policy_net(state).cpu()
            act = act.argmax(dim=1)
        self.policy_net.train()
        return act

    # pick samples randomly from replay memory (with batch_size)
    def train_policy_net(self, frame):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

        mini_batch = self.memory.sample_mini_batch(frame)
        mini_batch = np.array(mini_batch, dtype=object).transpose()

        history = np.stack(mini_batch[0], axis=0)
        states = np.float32(history[:, :4, :, :]) / 255.
        states = torch.from_numpy(states).to(device)
        actions = list(mini_batch[1])
        actions = torch.LongTensor(actions).to(device)
        rewards = list(mini_batch[2])
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = np.float32(history[:, 1:, :, :]) / 255.
        next_states = torch.from_numpy(next_states).to(device)
        dones = mini_batch[3] # checks if the game is over
        mask = torch.tensor(list(map(int, dones==False)),dtype=torch.uint8, device=device)
        
        # Your agent.py code here with double DQN modifications
        ### CODE ###
        s_t_a = self.policy_net(states).gather(dim=1, index=actions[:, None]).squeeze(1)
        # Compute Q function of next state
        # Find maximum Q-value of action at next state from policy net
        ### CODE ####
        # q_s_t_next = torch.empty(batch_size, device=device)
        # with torch.no_grad():
        #     q_s_t_next[mask] = self.target_net(next_states)[mask].max(dim=1)[0]

        q_s_t_next = self.target_net(next_states).gather(dim=1, index=self.policy_net(next_states).argmax(dim=1)[:, None])
        q_s_t_next = q_s_t_next * mask
        # Compute the Huber Loss
        ### CODE ####
        loss = torch.mean(((rewards + q_s_t_next * self.discount_factor).detach() - s_t_a) ** 2) 
        # only times discount factor once cuz r_t' where t' is already discounted by discount^(t' - t - 1)
        # Optimize the model, .step() both the optimizer and the scheduler!
        ### CODE ####
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.scheduler.step()
        