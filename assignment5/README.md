# Assignment 5: Deep Reinforcement Learning (DRL)

In this assignment, we train DRL agents to play the [Atari Breakout](https://www.gymlibrary.dev/environments/atari/breakout/) game. Specifically, we implement three kinds of RL agents, namely deep Q-network (DQN), double DQN, and DDPG agents. 

## DQN and DDQN
The action space of Breakout is enumerated as:

|Num|Action|
|:-:|---|
|0|No Op|
|1|Fire|
|2|Right|
|3|Left|

It is obviously a discrete action space, rendering DQN and DDQN inherently suitable for playing this game. Along with common training hacks that boost the performance of an RL agent, such as the replay buffer and epsilon-greedy policy, both DQN and DDQN demonstrate their abilities to master the game.

## DDPG
As a DDPG agent is designed for continuous action space, we treat the action vectors it predicts as an evaluation of each action. In other words, for Breakout, our DDPG agent outputs a 4-dimensional vector, and each entry of it represents a score corresponding to one of the four actions. At each step, we choose the action with the highest score.

## Result
Organizing...