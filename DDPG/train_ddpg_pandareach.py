import os
import time

import gymnasium as gym
import panda_gym            # registers PandaReach-v3
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
import matplotlib.pyplot as plt
import matplotlib

from gymnasium.spaces import flatten, flatten_space
from panda_gym import reward_type

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------------------
#  Replay Buffer
# ----------------------------
Transition = namedtuple("Transition", ("obs", "action", "reward", "next_obs", "done"))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, *args):
        self.buffer.append(Transition(*args))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))
    def __len__(self):
        return len(self.buffer)

# ----------------------------
#  Actor & Critic Nets
# ----------------------------
def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    bound = 1.0/np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-bound, bound)

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, act_dim)
        self.act_limit = act_limit
        # weight init
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # tanh to bound between -1 and 1, then scale
        return self.act_limit * torch.tanh(self.fc3(x))

class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fcs = nn.Linear(obs_dim, 400)
        self.fca = nn.Linear(act_dim, 300)
        self.fc2 = nn.Linear(400+300, 300)
        self.fc3 = nn.Linear(300, 1)
        # weight init
        self.fcs.weight.data   = fanin_init(self.fcs.weight.data.size())
        self.fca.weight.data   = fanin_init(self.fca.weight.data.size())
        self.fc2.weight.data   = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, obs, act):
        so = torch.relu(self.fcs(obs))
        sa = torch.relu(self.fca(act))
        x  = torch.relu(self.fc2(torch.cat([so, sa], dim=1)))
        return self.fc3(x)

# ----------------------------
#  DDPG Agent
# ----------------------------
class DDPGAgent:
    def __init__(
            self,
            obs_dim, act_dim, act_limit,
            actor_lr=1e-4, critic_lr=1e-3,
            gamma=0.99, tau=0.005,
            buffer_size=int(1e6), batch_size=256,
            noise_std=0.1,
            device=device
    ):
        self.device     = device
        self.gamma, self.tau = gamma, tau
        self.batch_size = batch_size
        self.act_limit  = act_limit

        # Replay
        self.replay = ReplayBuffer(buffer_size)

        # Networks
        self.actor       = Actor(obs_dim, act_dim, act_limit).to(device)
        self.critic      = Critic(obs_dim, act_dim).to(device)
        self.actor_tgt   = Actor(obs_dim, act_dim, act_limit).to(device)
        self.critic_tgt  = Critic(obs_dim, act_dim).to(device)
        self.actor_tgt.load_state_dict(self.actor.state_dict())
        self.critic_tgt.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.a_opt = optim.Adam(self.actor.parameters(),  lr=actor_lr)
        self.c_opt = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.noise_std = noise_std

    def select_action(self, obs, noise=True):
        obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            a = self.actor(obs).cpu().numpy()[0]
        if noise:
            a += np.random.normal(scale=self.noise_std, size=a.shape)
        return np.clip(a, -self.act_limit, self.act_limit)

    def store(self, *trans):
        self.replay.push(*trans)

    def update(self):
        if len(self.replay) < self.batch_size:
            return
        batch = self.replay.sample(self.batch_size)
        o      = torch.FloatTensor(np.vstack(batch.obs)).to(self.device)
        a      = torch.FloatTensor(np.vstack(batch.action)).to(self.device)
        r      = torch.FloatTensor(np.vstack(batch.reward)).to(self.device)
        no     = torch.FloatTensor(np.vstack(batch.next_obs)).to(self.device)
        d      = torch.FloatTensor(np.vstack(batch.done).astype(np.float32)).to(self.device)

        # Critic
        with torch.no_grad():
            a_tgt = self.actor_tgt(no)
            q_tgt = self.critic_tgt(no, a_tgt)
            y     = r + self.gamma * (1-d) * q_tgt
        q      = self.critic(o, a)
        c_loss = nn.MSELoss()(q, y)
        self.c_opt.zero_grad()
        c_loss.backward()
        self.c_opt.step()

        # Actor
        a_pred = self.actor(o)
        a_loss = -self.critic(o, a_pred).mean()
        self.a_opt.zero_grad()
        a_loss.backward()
        self.a_opt.step()

        # Soft updates
        for p, p_t in zip(self.actor.parameters(), self.actor_tgt.parameters()):
            p_t.data.mul_(1-self.tau)
            p_t.data.add_( self.tau * p.data)
        for p, p_t in zip(self.critic.parameters(), self.critic_tgt.parameters()):
            p_t.data.mul_(1-self.tau)
            p_t.data.add_( self.tau * p.data)

# ----------------------------
#  Training Loop
# ----------------------------
def train(env_name="PandaReach-v3", episodes=300, max_steps=200, render=False):
    env = gym.make(env_name, render_mode="human", reward_type="dense")

    # --- handle Dict observations:
    obs_space    = env.observation_space
    flat_obs_dim = int(flatten_space(obs_space).shape[0])
    act_dim      = env.action_space.shape[0]
    act_limit    = env.action_space.high[0]

    agent = DDPGAgent(flat_obs_dim, act_dim, act_limit, device=device)
    agent.actor.load_state_dict(torch.load("best_actor2.pth"))

    reward_hist = []
    best_avg    = -np.inf

    for ep in range(1, episodes+1):
        raw_obs, _ = env.reset()
        obs        = flatten(obs_space, raw_obs)
        ep_ret     = 0

        for t in range(max_steps):
            if render:
                env.render()
            a = agent.select_action(obs)
            raw_next, r, term, trunc, _ = env.step(a)
            next_obs   = flatten(obs_space, raw_next)
            done       = term or trunc

            agent.store(obs, a, r, next_obs, done)
            agent.update()

            obs     = next_obs
            ep_ret += r
            if done:
                break

        reward_hist.append(ep_ret)
        avg20 = np.mean(reward_hist[-20:])
        print(f"Ep {ep:3d}  Return: {ep_ret:7.3f}  Avg20: {avg20:7.3f}")

        if avg20 > best_avg:
            best_avg = avg20
            torch.save(agent.actor.state_dict(), "best_actor.pth")

    env.close()

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(reward_hist, label='Episode Reward')
    plt.plot(np.convolve(reward_hist, np.ones(20)/20, mode='valid'), label='20-Episode Avg')
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid()
    plt.savefig('training_progress8-max_steps-10.png')
    plt.show()
    print("⏹️ Training complete!")

def evaluate(env_name="PandaReach-v3", episodes=10, max_steps=200):
    env = gym.make(env_name, render_mode="human", reward_type="dense")
    obs_space = env.observation_space
    flat_obs_dim = int(flatten_space(obs_space).shape[0])
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]

    agent = DDPGAgent(flat_obs_dim, act_dim, act_limit, device=device)
    agent.actor.load_state_dict(torch.load("best_actor.pth"))

    for ep in range(1, episodes+1):
        raw_obs, _ = env.reset()
        obs        = flatten(obs_space, raw_obs)
        ep_ret     = 0

        for t in range(max_steps):
            time.sleep(0.1)
            env.render()
            a = agent.select_action(obs, noise=False)
            raw_next, r, term, trunc, _ = env.step(a)
            next_obs   = flatten(obs_space, raw_next)
            done       = term or trunc

            obs     = next_obs
            ep_ret += r
            if done:
                break

        print(f"Ep {ep:3d}  Return: {ep_ret:7.3f}")

    env.close()

if __name__ == "__main__":
    # train(episodes=600, max_steps=10, render=True)
    evaluate(episodes=100, max_steps=5)
