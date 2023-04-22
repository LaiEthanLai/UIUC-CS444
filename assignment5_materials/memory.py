from config import *
from collections import deque
import numpy as np
import random


class ReplayMemory(object):
    def __init__(self):
        self.memory = deque(maxlen=Memory_capacity)
    
    def push(self, history, action, reward, done):
        self.memory.append((history, action, reward, done))

    def sample_mini_batch(self, frame):
        mini_batch = []
        if frame >= Memory_capacity:
            sample_range = Memory_capacity
        else:
            sample_range = frame

        # history size
        sample_range -= (HISTORY_SIZE + 1)

        idx_sample = random.sample(range(sample_range), batch_size)
        for i in idx_sample:
            sample = []
            for j in range(HISTORY_SIZE + 1):
                sample.append(self.memory[i + j])

            sample = np.array(sample)
            mini_batch.append((np.stack(sample[:, 0], axis=0), sample[3, 1], sample[3, 2], sample[3, 3]))

        return mini_batch

    def __len__(self):
        return len(self.memory)


class ReplayMemoryLSTM(ReplayMemory):
    """
    This is a version of Replay Memory modified for LSTMs. 
    Replay memory generally stores (state, action, reward, next state).
    But LSTMs need sequential data. 
    So we store (state, action, reward, next state) for previous few states, constituting a trajectory.
    During training, the previous states will be used to generate the current state of LSTM. 
    Note that samples from previous episode might get included in the trajectory.
    Inspite of not being fully correct, this simple Replay Buffer performs well.
    """
    def __init__(self):
        super().__init__()

    def sample_mini_batch(self, frame):
        mini_batch = []
        if frame >= Memory_capacity:
            sample_range = Memory_capacity
        else:
            sample_range = frame

        sample_range -= (lstm_seq_length + 1)

        idx_sample = random.sample(range(sample_range - lstm_seq_length), batch_size)
        for i in idx_sample:
            sample = []
            for j in range(lstm_seq_length + 1):
                sample.append(self.memory[i + j])

            sample = np.array(sample)
            mini_batch.append((np.stack(sample[:, 0], axis=0), sample[lstm_seq_length - 1, 1], sample[lstm_seq_length - 1, 2], sample[lstm_seq_length - 1, 3]))

        return mini_batch
