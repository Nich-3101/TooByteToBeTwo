import os

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

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------------------------------------------------
#  Replay Buffer
#  Since TD3 is an off-policy algorithm, it learns from experiences and not
#  just the current policy's trajectory
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
# Weight initialization
# To help stabilize learning in deep networks by preventing
# exploding / vanishing gradients
# ---------------------------------------------------------
def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    bound = 1.0/np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-bound, bound)

# -----------------------------------------------------------------------------------
#  Deterministic Policy (Actor)
#  Difference with MBPO which uses Gaussian Policy is that TD3 uses a Gaussian Policy
# -----------------------------------------------------------------------------------
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
#  TD3 uses 2 Q-networks (Critic 1 and Critic 2) to apply Clipped Double Q-learning
# ---------------------------------------------------------------------------------
class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(256,256)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim+act_dim, hidden_sizes[0]), nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]), nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1)
        )
        # weight init
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
# ----------------
class TD3Agent:
    def __init__(self, obs_dim, act_dim, act_limit, device=device,
                 actor_lr=1e-3, critic_lr=1e-3, gamma=0.99, tau=0.005,
                 policy_noise=0.2, noise_clip=0.5, policy_delay=2,
                 buffer_size=int(1e6), batch_size=256):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.batch_size = batch_size
        self.total_it = 0

        # Replay buffer
        self.replay = ReplayBuffer(buffer_size)

        # Actor and Critic networks
        self.actor = DeterministicPolicy(obs_dim, act_dim, act_limit).to(device)
        self.actor_target = DeterministicPolicy(obs_dim, act_dim, act_limit).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic1 = QNetwork(obs_dim, act_dim).to(device)
        self.critic2 = QNetwork(obs_dim, act_dim).to(device)
        self.critic1_target = QNetwork(obs_dim, act_dim).to(device)
        self.critic2_target = QNetwork(obs_dim, act_dim).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr)

    def select_action(self, obs, noise=0.1):
        obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        action = self.actor(obs).cpu().data.numpy().flatten()
        if noise != 0:
            action = (action + np.random.normal(0, noise, size=action.shape))
        return np.clip(action, -self.actor.act_limit, self.actor.act_limit)

    def store(self, *args):
        self.replay.push(*args)

    def update(self):
        if len(self.replay) < self.batch_size:
            return

        self.total_it += 1

        # Sample batch
        batch = self.replay.sample(self.batch_size)
        obs = torch.FloatTensor(np.vstack(batch.obs)).to(self.device)
        act = torch.FloatTensor(np.vstack(batch.action)).to(self.device)
        rew = torch.FloatTensor(np.vstack(batch.reward)).to(self.device)
        next_obs = torch.FloatTensor(np.vstack(batch.next_obs)).to(self.device)
        done = torch.FloatTensor(np.vstack(batch.done).astype(np.float32)).to(self.device)

        # Select action according to policy and add clipped noise
        noise = (torch.randn_like(act) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
        next_action = (self.actor_target(next_obs) + noise).clamp(-self.actor.act_limit, self.actor.act_limit)

        # Compute target Q-value
        target_Q1 = self.critic1_target(next_obs, next_action)
        target_Q2 = self.critic2_target(next_obs, next_action)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = rew + (1 - done) * self.gamma * target_Q

        # Get current Q estimates
        current_Q1 = self.critic1(obs, act)
        current_Q2 = self.critic2(obs, act)

        # Compute critic loss
        critic1_loss = nn.MSELoss()(current_Q1, target_Q.detach())
        critic2_loss = nn.MSELoss()(current_Q2, target_Q.detach())

        # Optimize the critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_delay == 0:
            # Compute actor loss
            actor_loss = -self.critic1(obs, self.actor(obs)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor.pth")
        torch.save(self.critic1.state_dict(), filename + "_critic1.pth")
        torch.save(self.critic2.state_dict(), filename + "_critic2.pth")

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor.pth"))
        self.critic1.load_state_dict(torch.load(filename + "_critic1.pth"))
        self.critic2.load_state_dict(torch.load(filename + "_critic2.pth"))


def train(env_name="PandaReach-v3", episodes=300, max_steps=200, render=False):
    env = gym.make(env_name, render_mode="human", reward_type="dense")

    # --- handle Dict observations:
    obs_space    = env.observation_space
    flat_obs_dim = int(flatten_space(obs_space).shape[0])
    act_dim      = env.action_space.shape[0]
    act_limit    = env.action_space.high[0]

    agent = TD3Agent(flat_obs_dim, act_dim, act_limit, device=device)

    # Only load weights if the file exists
    if os.path.exists("best_actor.pth"):
        agent.actor.load_state_dict(torch.load("best_actor.pth"))
        print("Loaded pre-trained actor model from best_actor.pth")
    else:
        print("No pre-trained actor model found. Starting from scratch.")

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
    plt.savefig('training_progress1-max_steps-200.png')
    plt.show()
    print("⏹️ Training complete!")


# ----------------
#  Evaluation Loop
# ----------------
def evaluate(env_name="PandaReach-v3", episodes=10, max_steps=200):
    env = gym.make(env_name, render_mode="human", reward_type="dense")
    obs_space = env.observation_space
    flat_obs_dim = int(flatten_space(obs_space).shape[0])
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]

    agent = TD3Agent(flat_obs_dim, act_dim, act_limit)
    agent.load("best_policy.pth")

    for ep in range(1, episodes+1):
        raw_obs, _ = env.reset()
        obs = flatten(obs_space, raw_obs)
        ep_ret = 0
        for t in range(max_steps):
            #time.sleep(0.1)
            env.render()
            a = agent.select_action(obs, evaluate=True)
            raw_next, r, term, trunc, _ = env.step(a)
            obs = flatten(obs_space, raw_next)
            ep_ret += r
            if term or trunc:
                break
        print(f"Ep {ep:3d}  Return: {ep_ret:7.3f}")
    env.close()


if __name__ == "__main__":
    train(episodes=700, max_steps=50, render=True)
    #evaluate(episodes=100, max_steps=5)