import torch
import collections
import random

class ReplayBuffer:
    def __init__(self, buffer_limit=int(1e6), num_envs=1):
        self.buffer = collections.deque(maxlen=buffer_limit)
        self.num_envs = num_envs

    def push(self, obs, action, reward, next_obs, done):
        self.buffer.append(tuple([obs, action, reward, next_obs, done]))

    def sample(self, mini_batch_size):
        obs, action, reward, next_obs, done = zip(*random.sample(self.buffer, mini_batch_size))

        rand_idx = torch.randperm(mini_batch_size * self.num_envs)  # random shuffle tensors

        obs = torch.cat(obs)[rand_idx]
        action = torch.cat(action)[rand_idx]
        reward = torch.cat(reward)[rand_idx]
        next_obs = torch.cat(next_obs)[rand_idx]
        done = torch.cat(done)[rand_idx]
        return obs, action, reward, next_obs, done

    def size(self):
        return len(self.buffer)

import torch.distributions
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, num_obs=4, num_act=2):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_obs, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, num_act),
        )
    
    def forward(self, x):
        return self.net(x)


def soft_update(net, net_target, tau):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * tau + param.data *(1.0 - tau))

        
class DQN:
    def __init__(self, args):
        self.args = args

        self.env = Cartpole(args)
        self.replay = ReplayBuffer(num_envs=args.num_envs)

        self.act_space = 2
        self.discount = 0.99
        self.mini_batch_size = 128
        self.batch_size = self.args.num_envs * self.mini_batch_size
        self.tau = 0.995
        self.num_eval_freq = 100
        self.lr = 3e-4

        self.run_step = 1
        self.score = 0

        self.q        = Net(num_act=self.act_space).to(self.args.sim_device)
        self.q_target = Net(num_act=self.act_space).to(self.args.sim_device)

        soft_update(self.q, self.q_target, tau=0.0)

        self.q_target.eval()
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=self.lr)
    
    def update(self):
        self.optimizer.zero_grad()



from dqn import DQN
from ppo import PPO
from ppo_discrete import PPO_Discrete

import torch
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default=None, type=str)
parser.add_argument('--port', default=None, type=int)

parser.add_argument('--sim_device', type=str, default="cuda:0", help='Physics Device in PyTorch-like syntax')
parser.add_argument('--compute_device_id', default=0, type=int)
parser.add_argument('--graphics_device_id', type=int, default=0, help='Graphics Device ID')
parser.add_argument('--num_envs', default=512, type=int)
parser.add_argument('--headless', action='store_true')
parser.add_argument('--method', default='ppo', type=str)
parser.add_argument('--mode', default=None, type=str)
parser.add_argument('--port', default=None, type=int)

args = parser.parse_args()
args.headless = True

torch.manual_seed(0)
random.seed(0)

if args.method == 'ppo':
    policy = PPO(args)
elif args.method == 'dqn':
    policy = PPO_Discrete(args)
elif args.method == 'dqn':
    policy = DQN(args)

while True:
    policy.run()
