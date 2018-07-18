#!/usr/bin/env python3
import gym
import collections
from tensorboardX import SummaryWriter
import numpy as np

ENV_NAME = "FrozenLake-v0"
GAMMA = 0.9
TEST_EPISODES = 100
ALPHA = 0.2

class Agent:
    def __init__(self):
        self.values = collections.defaultdict(float)
        self.state_transitions = collections.defaultdict(collections.Counter)
        self.rewards = collections.defaultdict(float)
        self.env = gym.make(ENV_NAME)
        self.state = self.env.reset()

    def random_play_n_steps(self, n):
        result = []
        for i in range(n):
            action = self.env.action_space.sample()
            next_state, reward, done, _ = self.env.step(action)
            sample = (self.state, action, reward, next_state)
            result.append(sample)
            self.state_transitions[(self.state, action)][next_state]+=1
            self.rewards[(self.state, action, next_state)] = reward
            if done:
                self.state = self.env.reset()
            else:
                self.state = next_state
        return result

    def action(self, state):
        action_values = []
        for action in range(self.env.action_space.n):
            action_values.append((self.values[(state, action)], action))
        return max(action_values)[1]

    def play_episode(self, env):
        state = self.env.reset()
        done = False
        while not done:
            action = self.action(state)
            next_state, reward, done, _ = self.env.step(action)
            self.state_transitions[(state, action)][next_state]+=1
            self.rewards[(state, action, next_state)] = reward
            state = next_state
            if done:
                return reward

    def update_values(self, samples):
        for state, action, reward, next_state in samples:
            self.values[(state, action)] = (1-ALPHA)*self.values[(state, action)] + ALPHA*(reward + GAMMA*self.values[(next_state, self.action(next_state))])

if __name__ == "__main__":
    test_env = gym.make(ENV_NAME)
    agent = Agent()
    i = 0
    while True:
        samples = agent.random_play_n_steps(100)
        agent.update_values(samples)
        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
        reward /= TEST_EPISODES
        print("episode %d, mean: %f" % (i, reward))
        if reward > 0.8:
            print("solved!")
            break
        i+=1
