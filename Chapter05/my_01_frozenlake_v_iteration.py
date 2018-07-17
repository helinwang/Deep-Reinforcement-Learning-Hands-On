#!/usr/bin/env python3
import gym
import collections
from tensorboardX import SummaryWriter
import numpy as np

ENV_NAME = "FrozenLake-v0"
GAMMA = 0.99
TEST_EPISODES = 100

class Agent:
    def __init__(self):
        self.values = collections.defaultdict(float)
        self.state_transitions = collections.defaultdict(collections.Counter)
        self.rewards = collections.defaultdict(float)
        self.env = gym.make(ENV_NAME)

    def play_n_random_steps(self, n):
        state = self.env.reset()
        for i in range(n):
            action = self.env.action_space.sample()
            next_state, reward, done, _ = self.env.step(action)
            self.state_transitions[(state, action)][next_state]+=1
            self.rewards[(state, action, next_state)] = reward
            if done:
                state = self.env.reset()
            else:
                state = next_state

    def calc_action_value(self, state, action):
        total = sum(self.state_transitions[(state, action)].values())
        value = 0.0
        for next_state, count in self.state_transitions[(state, action)].items():
            prob = count / total
            value += prob * (self.rewards[(state, action, next_state)] + GAMMA*self.values[next_state])
        return value

    def action(self, state):
        action_values = []
        for action in range(self.env.action_space.n):
            action_values.append((self.calc_action_value(state, action), action))
        return max(action_values)[1]

    def play_episode(self):
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

    def update_values(self):
        for state in range(self.env.observation_space.n):
            values = []
            for action in range(self.env.action_space.n):
                values.append((self.calc_action_value(state, action), action))
            self.values[state] = max(values)[0]

if __name__ == "__main__":
    agent = Agent()
    i = 0
    while True:
        agent.play_n_random_steps(100)
        agent.update_values()
        reward = 0.0

        for _ in range(TEST_EPISODES):
            reward += agent.play_episode()
        reward /= TEST_EPISODES
        print("episode %d, mean: %f" % (i, reward))
        if reward > 0.8:
            print("solved!")
            break
        i+=1
