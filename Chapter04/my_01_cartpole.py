#!/usr/bin/env python3
import gym
import torch.nn as nn
import torch
import torch.optim as optim
from collections import namedtuple
import numpy as np

HIDDEN_SIZE = 64
BATCH_SIZE = 16

EpiStep = namedtuple("EpiStep", field_names=["state", "action", "reward"])
Episode = namedtuple("Episode", field_names=["gain", "steps"])

class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, action_size):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, x):
        return self.net(x)

def infer(net, softmax, obs):
    actions = net(torch.Tensor(obs))
    actions = softmax(actions)
    probs = actions.data.numpy()
    return np.random.choice(len(probs), p=probs)

def iter_batch(env, net, batch_size):
    while True:
        ret = []
        for i in range(batch_size):
            obs = env.reset()
            done = False
            gain = 0
            steps = []
            while not done:
                action = infer(net, nn.Softmax(dim=0), obs)
                new_obs, reward, done, _ = env.step(action)
                epi_step = EpiStep(obs, action, reward)
                gain += reward
                steps.append(epi_step)
                obs = new_obs
            ret.append(Episode(gain, steps))
        yield ret

def filter_batch(batch):
    gains = [e.gain for e in batch]
    cutoff = np.percentile(gains, 70)
    obs = []
    actions = []
    rewards = []
    for e in batch:
        if e.gain < cutoff:
            continue
        obs.extend([x.state for x in e.steps])
        actions.extend([x.action for x in e.steps])
        rewards.append(e.gain)
    return obs, actions, np.mean(rewards), cutoff

if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    env = gym.wrappers.Monitor(env, directory="mon", force=True)
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n
    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    loss = nn.CrossEntropyLoss()
    adam = optim.Adam(params=net.parameters(), lr=0.01)
    for i, batch in enumerate(iter_batch(env, net, BATCH_SIZE)):
        obs_v, acts_v, reward_m, reward_b = filter_batch(batch)
        adam.zero_grad()
        scores_v = net(torch.Tensor(obs_v))
        loss_v = loss(scores_v, torch.LongTensor(acts_v))
        loss_v.backward()
        adam.step()
        print("%d: loss=%.3f, reward_mean=%.1f, reward_bound=%.1f" % (
            i, loss_v.item(), reward_m, reward_b))
        if reward_m > 199:
            print("Solved!")
            break
