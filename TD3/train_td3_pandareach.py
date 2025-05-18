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
from gymnasium.spaces import flatten, flatten_space
from torch.utils.tensorboard import SummaryWriter

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------------------------------------------------
#  Replay Buffer
# -------------------------------------------------------------------------
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

# ---------------------------------------------------------
# Weight initialization helper
def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    bound = 1.0/np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-bound, bound)

# -----------------------------------------------------------------------------------
#  Deterministic Policy (Actor)
class DeterministicPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit, hidden_sizes=(256, 256)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_sizes[0]), nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]), nn.ReLU(),
            nn.Linear(hidden_sizes[1], act_dim), nn.Tanh()
        )
        self.act_limit = act_limit

    def forward(self, obs):
        return self.net(obs) * self.act_limit

# ---------------------------------------------------------------------------------
#  Q-Network (Critic)
class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(256,256)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim+act_dim, hidden_sizes[0]), nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]), nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1)
        )
        for m in self.net:
            if isinstance(m, nn.Linear) and m.out_features != 1:
                m.weight.data = fanin_init(m.weight.data.size())
            if isinstance(m, nn.Linear) and m.out_features == 1:
                m.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=1)
        return self.net(x)

# ----------------
#  TD3 Agent Class
class TD3Agent:
    def __init__(self, obs_dim, act_dim, act_limit, writer: SummaryWriter, **kwargs):
        self.device = device
        self.gamma = kwargs.get('gamma', 0.99)
        self.tau = kwargs.get('tau', 0.005)
        self.policy_noise = kwargs.get('policy_noise', 0.2)
        self.noise_clip = kwargs.get('noise_clip', 0.5)
        self.policy_delay = kwargs.get('policy_delay', 2)
        self.batch_size = kwargs.get('batch_size', 256)
        self.total_it = 0
        self.writer = writer

        # Replay buffer
        self.replay = ReplayBuffer(kwargs.get('buffer_size', int(1e6)))

        # Actor/critic
        self.actor = DeterministicPolicy(obs_dim, act_dim, act_limit).to(device)
        self.actor_target = DeterministicPolicy(obs_dim, act_dim, act_limit).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1 = QNetwork(obs_dim, act_dim).to(device)
        self.critic2 = QNetwork(obs_dim, act_dim).to(device)
        self.critic1_target = QNetwork(obs_dim, act_dim).to(device)
        self.critic2_target = QNetwork(obs_dim, act_dim).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # Opt
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=kwargs.get('actor_lr',1e-3))
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=kwargs.get('critic_lr',1e-3))
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=kwargs.get('critic_lr',1e-3))

    def select_action(self, obs, noise=0.1):
        obs = torch.FloatTensor(obs).unsqueeze(0).to(device)
        action = self.actor(obs).cpu().data.numpy().flatten()
        if noise != 0:
            action += np.random.normal(0, noise, size=action.shape)
        return np.clip(action, -self.actor.act_limit, self.actor.act_limit)

    def store(self, *args): self.replay.push(*args)

    def update(self):
        if len(self.replay) < self.batch_size:
            return
        self.total_it += 1
        batch = self.replay.sample(self.batch_size)
        obs = torch.FloatTensor(np.vstack(batch.obs)).to(device)
        act = torch.FloatTensor(np.vstack(batch.action)).to(device)
        rew = torch.FloatTensor(np.vstack(batch.reward)).to(device)
        next_obs = torch.FloatTensor(np.vstack(batch.next_obs)).to(device)
        done = torch.FloatTensor(np.vstack(batch.done).astype(np.float32)).to(device)

        # target actions with noise
        noise = (torch.randn_like(act) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
        next_act = (self.actor_target(next_obs) + noise).clamp(-self.actor.act_limit, self.actor.act_limit)

        # target Q
        target_Q = torch.min(
            self.critic1_target(next_obs, next_act),
            self.critic2_target(next_obs, next_act)
        )
        target_Q = rew + (1 - done) * self.gamma * target_Q

        # current Q
        current_Q1 = self.critic1(obs, act)
        current_Q2 = self.critic2(obs, act)

        # critic loss
        critic1_loss = nn.MSELoss()(current_Q1, target_Q.detach())
        critic2_loss = nn.MSELoss()(current_Q2, target_Q.detach())

        # optimize critics
        self.critic1_optimizer.zero_grad(); critic1_loss.backward(); self.critic1_optimizer.step()
        self.critic2_optimizer.zero_grad(); critic2_loss.backward(); self.critic2_optimizer.step()

        # log critic Q-values
        self.writer.add_scalar('Critic/Q1', current_Q1.mean().item(), self.total_it)
        self.writer.add_scalar('Critic/Q2', current_Q2.mean().item(), self.total_it)

        # log target Q-values
        self.writer.add_scalar('Critic/TargetQ1', self.critic1_target.mean().item(), self.total_it)
        self.writer.add_scalar('Critic/TargetQ2', self.critic2_target.mean().item(), self.total_it)

        # log critic losses
        self.writer.add_scalar('Loss/critic1', critic1_loss.item(), self.total_it)
        self.writer.add_scalar('Loss/critic2', critic2_loss.item(), self.total_it)

        # l
        if self.total_it % self.policy_delay == 0:
            # actor loss
            actor_loss = -self.critic1(obs, self.actor(obs)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self.writer.add_scalar('Loss/actor', actor_loss.item(), self.total_it)

            # log actor value
            self.writer.add_scalar('Actor/Value', self.critic1(obs, self.actor(obs)).mean().item(), self.total_it)

            # update targets
            for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for p, tp in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for p, tp in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename))
        self.critic1.load_state_dict(torch.load(filename))
        self.critic2.load_state_dict(torch.load(filename))


def train(env_name="PandaReach-v3", episodes=300, max_steps=100, render=False):
    writer = SummaryWriter(log_dir="runs/TD3")
    start = time.time()

    if render:
        env = gym.make(env_name, render_mode="human", reward_type="dense")
    else:
        env = gym.make(env_name, reward_type="dense")

    obs_space = env.observation_space
    flat_dim = int(flatten_space(obs_space).shape[0])
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]

    step = 0
    agent = TD3Agent(flat_dim, act_dim, act_limit, writer)
    if os.path.exists("best_actor.pth"):
        agent.actor.load_state_dict(torch.load("best_actor.pth", map_location=device))
        print("Loaded pre-trained actor.")

    reward_hist, best_avg = [], -np.inf
    for ep in range(1, episodes+1):
        time_start = time.time()
        raw, _ = env.reset()
        obs = flatten(obs_space, raw)
        ep_ret = 0
        for t in range(max_steps):
            if render: env.render()
            a = agent.select_action(obs)
            raw_n, r, term, trunc, _ = env.step(a)
            next_obs = flatten(obs_space, raw_n)
            done = term or trunc
            agent.store(obs, a, r, next_obs, done)
            agent.update()

            step += 1

            obs = next_obs; ep_ret += r

            # log rewards per step
            writer.add_scalar('Reward/step', r, step)
            if done: break

        reward_hist.append(ep_ret)
        avg20 = np.mean(reward_hist[-20:])
        time_end = time.time()
        print(f"Ep {ep:3d}  Return: {ep_ret:7.3f}  Avg20: {avg20:7.3f}")

        # log rewards
        writer.add_scalar('Time/episode', time_end - time_start, ep)
        writer.add_scalar('Reward/episode', ep_ret, ep)
        writer.add_scalar('Reward/avg20', avg20, ep)

        if avg20 > best_avg:
            best_avg = avg20
            torch.save(agent.actor.state_dict(), "best_actor.pth")

    duration = time.time() - start
    mins, secs = divmod(duration, 60)
    print(f"⏱️ Total time: {int(mins)}m {int(secs)}s")

    # close writer
    writer.close()

    env.close()
    # Plot offline for quick view
    plt.figure(figsize=(10,5))
    plt.plot(reward_hist, label='Episode Reward')
    plt.plot(np.convolve(reward_hist, np.ones(20)/20, mode='valid'), label='20-Episode Avg')
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend(); plt.grid()
    plt.savefig('sac_training.png')
    plt.show()
    print("⏹️ Training complete!")


def evaluate(env_name="PandaReach-v3", episodes=10, max_steps=200):
    env = gym.make(env_name, render_mode="human", reward_type="dense")
    obs_space = env.observation_space
    flat_dim = int(flatten_space(obs_space).shape[0])
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]
    agent = TD3Agent(flat_dim, act_dim, act_limit, SummaryWriter())
    agent.actor.load_state_dict(torch.load("best_actor.pth"))
    for ep in range(1, episodes+1):
        raw, _ = env.reset(); obs = flatten(obs_space, raw)
        ep_ret = 0
        for t in range(max_steps):
            time.sleep(0.1); env.render()
            a = agent.select_action(obs, noise=0)
            raw_n, r, term, trunc, _ = env.step(a)
            obs = flatten(obs_space, raw_n)
            ep_ret += r
            if term or trunc: break
        print(f"Ep {ep:3d}  Return: {ep_ret:7.3f}")
    env.close()

if __name__ == "__main__":
    train(episodes=1000, max_steps=50, render=False)
    # evaluate(episodes=5, max_steps=5)
