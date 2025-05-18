import os
import time

import gymnasium as gym
import panda_gym            # registers PandaReach-v3
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import matplotlib.pyplot as plt
import matplotlib

from gymnasium.spaces import flatten, flatten_space
from panda_gym import reward_type

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------------------
#  PPO Trajectory Buffer
# ----------------------------
PPOTransition = namedtuple("PPOTransition", ("obs", "action", "logprob", "value", "reward", "done"))

class PPOBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(PPOTransition(*args))

    def get_all(self):
        batch = PPOTransition(*zip(*self.buffer))
        return batch

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)

# ----------------------------
#  Actor & Critic Nets
# ----------------------------
def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    bound = 1.0/np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-bound, bound)

class PolicyActor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.mean = nn.Linear(300, act_dim)
        self.log_std = nn.Linear(300, act_dim)
        self.act_limit = act_limit

        # Weight initialization
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.mean.weight.data.uniform_(-3e-3, 3e-3)
        self.log_std.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)  # Clamp for numerical stability
        std = torch.exp(log_std)
        return mean, std

    def get_action(self, obs, deterministic=False):
        mean, std = self.forward(obs)

        if deterministic:
            # During evaluation, just use the mean action
            action = mean
        else:
            # During training, sample from the distribution
            normal = torch.distributions.Normal(mean, std)
            action = normal.sample()

        action = torch.clamp(action, -self.act_limit, self.act_limit)

        # Also calculate log probability for the PPO objective
        if not deterministic:
            # We need to calculate log prob before clamping
            log_prob = normal.log_prob(action).sum(dim=-1, keepdim=True)
        else:
            log_prob = None

        return action, log_prob

class ValueCritic(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

        # Weight initialization
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# ----------------------------
#  PPO Agent
# ----------------------------
class PPOAgent:
    def __init__(
            self,
            obs_dim, act_dim, act_limit,
            actor_lr=3e-4, critic_lr=1e-3,
            gamma=0.99, gae_lambda=0.95,
            clip_ratio=0.2, entropy_coef=0.01,
            value_coef=0.5, max_grad_norm=0.5,
            buffer_size=2048, batch_size=64,
            update_epochs=10,
            device=device
    ):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.update_epochs = update_epochs
        self.act_limit = act_limit

        # Buffer
        self.buffer = PPOBuffer(buffer_size)

        # Networks
        self.actor = PolicyActor(obs_dim, act_dim, act_limit).to(device)
        self.critic = ValueCritic(obs_dim).to(device)

        # Optimizers
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr)

    def select_action(self, obs, deterministic=False):
        """Select action given observation"""
        obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob = self.actor.get_action(obs, deterministic)
            value = self.critic(obs)

        return (
            action.cpu().numpy()[0],
            log_prob.cpu().numpy()[0] if log_prob is not None else None,
            value.cpu().numpy()[0]
        )

    def store(self, *transition):
        """Store transition in buffer"""
        self.buffer.push(*transition)

    def compute_advantages(self, values, rewards, dones, last_value):
        """Compute GAE advantages and returns"""
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)

        last_gae = 0
        last_return = last_value

        for t in reversed(range(len(rewards))):
            # For the last state, we need to bootstrap with the last value
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]

            # Compute TD error (delta)
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]

            # Compute GAE advantage
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae

            # Compute returns for value function target
            returns[t] = last_return = rewards[t] + self.gamma * next_non_terminal * last_return

        return advantages, returns

    def update(self):
        """Update policy and value function"""
        if len(self.buffer) < self.batch_size:
            return

        # Get all data from buffer
        batch = self.buffer.get_all()

        # Convert to tensors
        states = torch.FloatTensor(np.vstack(batch.obs)).to(self.device)
        actions = torch.FloatTensor(np.vstack(batch.action)).to(self.device)
        old_log_probs = torch.FloatTensor(np.vstack(batch.logprob)).to(self.device)
        old_values = torch.FloatTensor(np.vstack(batch.value)).to(self.device)
        rewards = np.vstack(batch.reward)
        dones = np.vstack(batch.done)

        # Compute advantages and returns
        with torch.no_grad():
            last_obs = torch.FloatTensor(batch.obs[-1]).unsqueeze(0).to(self.device)
            last_value = self.critic(last_obs).cpu().numpy()[0]

        advantages, returns = self.compute_advantages(old_values.cpu().numpy(), rewards, dones, last_value)

        # Convert to tensors
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Prepare for minibatches
        batch_size = len(self.buffer)
        num_batches = max(batch_size // self.batch_size, 1)

        # PPO update epochs
        for _ in range(self.update_epochs):
            # Create random indices
            indices = np.random.permutation(batch_size)

            # Update in minibatches
            for start in range(0, batch_size, self.batch_size):
                end = min(start + self.batch_size, batch_size)
                batch_indices = indices[start:end]

                # Get minibatch data
                mb_states = states[batch_indices]
                mb_actions = actions[batch_indices]
                mb_old_log_probs = old_log_probs[batch_indices]
                mb_advantages = advantages[batch_indices]
                mb_returns = returns[batch_indices]

                # Get current policy outputs
                _, std = self.actor(mb_states)
                dist = torch.distributions.Normal(
                    self.actor.mean(self.actor.fc2(F.relu(self.actor.fc1(mb_states)))),
                    std
                )
                current_log_probs = dist.log_prob(mb_actions).sum(dim=-1, keepdim=True)
                entropy = dist.entropy().sum(dim=-1, keepdim=True)

                # Get current value estimates
                current_values = self.critic(mb_states)

                # PPO policy loss
                ratio = torch.exp(current_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Entropy bonus
                entropy_loss = -self.entropy_coef * entropy.mean()

                # Value loss
                value_loss = self.value_coef * F.mse_loss(current_values, mb_returns)

                # Total loss
                total_loss = policy_loss + entropy_loss + value_loss

                # Perform update
                self.actor_opt.zero_grad()
                self.critic_opt.zero_grad()
                total_loss.backward()

                # Clip gradients
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

                self.actor_opt.step()
                self.critic_opt.step()

        # Clear buffer after update
        self.buffer.clear()

    def save(self, path="best_actor_ppo.pth", critic_path="best_critic_ppo.pth"):
        """Save actor and critic models"""
        torch.save(self.actor.state_dict(), path)
        torch.save(self.critic.state_dict(), critic_path)

    def load(self, path="best_actor_ppo.pth", critic_path="best_critic_ppo.pth"):
        """Load actor and critic models"""
        self.actor.load_state_dict(torch.load(path, map_location=self.device))
        try:
            self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
        except:
            print("Could not load critic model")

# ----------------------------
#  Training Loop
# ----------------------------
def train(env_name="PandaReach-v3", episodes=300, steps_per_epoch=2048, render=False):
    env = gym.make(env_name, render_mode="human" if render else None, reward_type="dense")

    # --- handle Dict observations:
    obs_space = env.observation_space
    flat_obs_dim = int(flatten_space(obs_space).shape[0])
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]

    agent = PPOAgent(flat_obs_dim, act_dim, act_limit, device=device)

    try:
        agent.load()
        print("Loaded existing policy")
    except:
        print("Starting with new policy")

    reward_hist = []
    epoch_rewards = []
    best_avg = -np.inf

    total_steps = 0
    epoch = 0

    while total_steps < episodes * steps_per_epoch:
        epoch += 1
        raw_obs, _ = env.reset()
        obs = flatten(obs_space, raw_obs)
        epoch_steps = 0
        ep_reward = 0

        # Collect trajectory
        while epoch_steps < steps_per_epoch:
            if render:
                env.render()

            # Select action
            action, log_prob, value = agent.select_action(obs)

            # Take action
            raw_next, reward, term, trunc, _ = env.step(action)
            next_obs = flatten(obs_space, raw_next)
            done = term or trunc

            # Store transition
            agent.store(obs, action, log_prob, value, reward, done)

            # Update counters
            epoch_steps += 1
            total_steps += 1
            ep_reward += reward

            # Move to next state
            obs = next_obs

            # If episode ended, reset
            if done:
                epoch_rewards.append(ep_reward)
                raw_obs, _ = env.reset()
                obs = flatten(obs_space, raw_obs)
                ep_reward = 0

            # If we've collected enough steps, update policy
            if epoch_steps >= steps_per_epoch:
                break

        # Update policy after collecting trajectory
        agent.update()

        # Record average reward for this epoch
        avg_epoch_reward = np.mean(epoch_rewards) if epoch_rewards else 0
        reward_hist.append(avg_epoch_reward)
        epoch_rewards = []

        # Calculate running average
        avg20 = np.mean(reward_hist[-20:]) if len(reward_hist) >= 20 else np.mean(reward_hist)

        print(f"Epoch {epoch:3d}  AvgReturn: {avg_epoch_reward:7.3f}  Avg20: {avg20:7.3f}  Steps: {total_steps}")

        # Save best policy
        if avg20 > best_avg:
            best_avg = avg20
            agent.save()

    env.close()

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(reward_hist, label='Epoch Reward')
    plt.plot(np.convolve(reward_hist, np.ones(20)/20, mode='valid'), label='20-Epoch Avg') if len(reward_hist) >= 20 else None
    plt.title('PPO Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.grid()
    plt.savefig('ppo_training_progress.png')
    plt.show()
    print("⏹️ PPO Training complete!")

# ----------------------------
#  Evaluation Loop
# ----------------------------
def evaluate(env_name="PandaReach-v3", episodes=10, max_steps=200):
    env = gym.make(env_name, render_mode="human", reward_type="dense")

    obs_space = env.observation_space
    flat_obs_dim = int(flatten_space(obs_space).shape[0])
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]

    agent = PPOAgent(flat_obs_dim, act_dim, act_limit, device=device)
    agent.load()

    for ep in range(1, episodes+1):
        raw_obs, _ = env.reset()
        obs = flatten(obs_space, raw_obs)
        ep_ret = 0

        for t in range(max_steps):
            time.sleep(0.1)
            env.render()

            # Select deterministic action for evaluation
            action, _, _ = agent.select_action(obs, deterministic=True)

            raw_next, r, term, trunc, _ = env.step(action)
            next_obs = flatten(obs_space, raw_next)
            done = term or trunc

            obs = next_obs
            ep_ret += r

            if done:
                break

        print(f"Ep {ep:3d}  Return: {ep_ret:7.3f}")

    env.close()

if __name__ == "__main__":
    # train(episodes=1000, steps_per_epoch=2048, render=True)
    evaluate(episodes=10, max_steps=200)