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

# Device
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
#  Gaussian Policy (Actor)
# ----------------------------
LOG_STD_MAX = 2
LOG_STD_MIN = -20

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    bound = 1.0/np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-bound, bound)

class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit, hidden_sizes=(256,256)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_sizes[0]), nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]), nn.ReLU()
        )
        self.mean_linear = nn.Linear(hidden_sizes[1], act_dim)
        self.log_std_linear = nn.Linear(hidden_sizes[1], act_dim)
        self.act_limit = act_limit

        # weight init
        for m in self.net:
            if isinstance(m, nn.Linear):
                m.weight.data = fanin_init(m.weight.data.size())
        self.mean_linear.weight.data.uniform_(-3e-3, 3e-3)
        self.log_std_linear.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, obs):
        x = self.net(obs)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        return mean, std

    def sample(self, obs):
        mean, std = self.forward(obs)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # reparameterization
        y_t = torch.tanh(x_t)
        action = y_t * self.act_limit
        log_prob = normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

# ----------------------------
#  Q-Network (Critic)
# ----------------------------
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


# Add this class above SACAgent
class DynamicsModel(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(256, 256)):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_sizes[0]), nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]), nn.ReLU(),
            nn.Linear(hidden_sizes[1], obs_dim)  # predict next_obs
        )
    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=1)
        return self.model(x)

def train_dynamics_model(model, replay, batch_size=256, epochs=5):
    if len(replay) < batch_size:
        return
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        batch = replay.sample(batch_size)
        obs = torch.FloatTensor(np.vstack(batch.obs)).to(device)
        act = torch.FloatTensor(np.vstack(batch.action)).to(device)
        next_obs = torch.FloatTensor(np.vstack(batch.next_obs)).to(device)
        pred_next = model(obs, act)
        loss = loss_fn(pred_next, next_obs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def generate_model_rollouts(model, policy, replay, rollout_length=1, batch_size=128):
    if len(replay) < batch_size:
        return []
    batch = replay.sample(batch_size)
    obs = torch.FloatTensor(np.vstack(batch.obs)).to(device)
    synthetic = []
    for _ in range(rollout_length):
        with torch.no_grad():
            act = torch.FloatTensor([policy.select_action(o.cpu().numpy()) for o in obs]).to(device)
            next_obs = model(obs, act)
        reward = torch.zeros((batch_size, 1))
        done = torch.zeros((batch_size, 1))
        for i in range(batch_size):
            synthetic.append((obs[i].cpu().numpy(), act[i].cpu().numpy(),
                              reward[i].item(), next_obs[i].cpu().numpy(), done[i].item()))
        obs = next_obs
    return synthetic

# ----------
#  SAC Agent
# ----------
class SACAgent:
    def __init__(self,
                 obs_dim, act_dim, act_limit,
                 actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4,
                 gamma=0.99, tau=0.005,
                 buffer_size=int(1e6), batch_size=256,
                 target_entropy=None,
                 device=device):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        # Replay buffer
        self.replay = ReplayBuffer(buffer_size)

        # Networks
        self.policy = GaussianPolicy(obs_dim, act_dim, act_limit).to(device)
        self.q1 = QNetwork(obs_dim, act_dim).to(device)
        self.q2 = QNetwork(obs_dim, act_dim).to(device)
        self.q1_tgt = QNetwork(obs_dim, act_dim).to(device)
        self.q2_tgt = QNetwork(obs_dim, act_dim).to(device)
        self.q1_tgt.load_state_dict(self.q1.state_dict())
        self.q2_tgt.load_state_dict(self.q2.state_dict())

        # Entropy temperature
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.target_entropy = -act_dim if target_entropy is None else target_entropy

        # Optimizers
        self.policy_opt = optim.Adam(self.policy.parameters(), lr=actor_lr)
        self.q1_opt = optim.Adam(self.q1.parameters(), lr=critic_lr)
        self.q2_opt = optim.Adam(self.q2.parameters(), lr=critic_lr)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=alpha_lr)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs, evaluate=False):
        obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        if evaluate:
            mean, _ = self.policy.forward(obs)
            action = torch.tanh(mean) * self.policy.act_limit
            return action.cpu().detach().numpy()[0]
        action, _ = self.policy.sample(obs)
        return action.cpu().detach().numpy()[0]

    def store(self, *trans):
        self.replay.push(*trans)

    def update(self):
        if len(self.replay) < self.batch_size:
            return
        # Sample batch
        batch = self.replay.sample(self.batch_size)
        obs = torch.FloatTensor(np.vstack(batch.obs)).to(self.device)
        act = torch.FloatTensor(np.vstack(batch.action)).to(self.device)
        rew = torch.FloatTensor(np.vstack(batch.reward)).to(self.device)
        next_obs = torch.FloatTensor(np.vstack(batch.next_obs)).to(self.device)
        done = torch.FloatTensor(np.vstack(batch.done).astype(np.float32)).to(self.device)

        # Critic update
        with torch.no_grad():
            next_act, next_logp = self.policy.sample(next_obs)
            q1_t = self.q1_tgt(next_obs, next_act)
            q2_t = self.q2_tgt(next_obs, next_act)
            q_t_min = torch.min(q1_t, q2_t) - self.alpha * next_logp
            y = rew + self.gamma * (1 - done) * q_t_min

        q1_val = self.q1(obs, act)
        q2_val = self.q2(obs, act)
        q1_loss = nn.MSELoss()(q1_val, y)
        q2_loss = nn.MSELoss()(q2_val, y)
        self.q1_opt.zero_grad(); q1_loss.backward(); self.q1_opt.step()
        self.q2_opt.zero_grad(); q2_loss.backward(); self.q2_opt.step()

        # Policy update
        new_act, logp = self.policy.sample(obs)
        q1_new = self.q1(obs, new_act)
        q2_new = self.q2(obs, new_act)
        q_new_min = torch.min(q1_new, q2_new)
        policy_loss = (self.alpha * logp - q_new_min).mean()
        self.policy_opt.zero_grad(); policy_loss.backward(); self.policy_opt.step()

        # Alpha update
        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad(); alpha_loss.backward(); self.alpha_opt.step()

        # Soft update targets
        for p, p_t in zip(self.q1.parameters(), self.q1_tgt.parameters()):
            p_t.data.mul_(1-self.tau); p_t.data.add_(self.tau * p.data)
        for p, p_t in zip(self.q2.parameters(), self.q2_tgt.parameters()):
            p_t.data.mul_(1-self.tau); p_t.data.add_(self.tau * p.data)

    def save(self, path="best_policy.pth"):
        torch.save(self.policy.state_dict(), path)

    def load(self, path="best_policy.pth"):
        self.policy.load_state_dict(torch.load(path, map_location=torch.device(device)))


# ----------------------------
#  Training Loop
# ----------------------------
def train(env_name="PandaReach-v3", episodes=300, max_steps=200, render=False):
    # Start timing
    start_time = time.time()

    env = gym.make(env_name, render_mode="human", reward_type="dense")
    obs_space = env.observation_space
    flat_obs_dim = int(flatten_space(obs_space).shape[0])
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]

    agent = SACAgent(flat_obs_dim, act_dim, act_limit)
    agent.load("best_policy.pth")
    dynamics_model = DynamicsModel(flat_obs_dim, act_dim).to(device)

    reward_hist = []
    best_avg = -np.inf

    for ep in range(1, episodes+1):
        raw_obs, _ = env.reset()
        obs = flatten(obs_space, raw_obs)
        ep_ret = 0

        for t in range(max_steps):
            if render:
                env.render()
            a = agent.select_action(obs)
            raw_next, r, term, trunc, _ = env.step(a)
            next_obs = flatten(obs_space, raw_next)
            done = term or trunc

            agent.store(obs, a, r, next_obs, done)
            agent.update()

            obs = next_obs
            ep_ret += r
            if done:
                break

        # Train dynamics model on real experience
        train_dynamics_model(dynamics_model, agent.replay, batch_size=256, epochs=5)

        # Generate synthetic rollouts using the learned model
        synthetic_transitions = generate_model_rollouts(dynamics_model, agent, agent.replay, rollout_length=1)
        for trans in synthetic_transitions:
            agent.store(*trans)

        reward_hist.append(ep_ret)
        avg20 = np.mean(reward_hist[-20:])
        print(f"Ep {ep:3d}  Return: {ep_ret:7.3f}  Avg20: {avg20:7.3f}")
        if avg20 > best_avg:
            best_avg = avg20
            agent.save("best_policy.pth")

    env.close()

    # End timing
    end_time = time.time()
    training_time = end_time - start_time
    mins, secs = divmod(training_time, 60)
    print(f"\n⏱️ Total training time: {int(mins)} minutes {int(secs)} seconds")

    plt.figure(figsize=(10, 5))
    plt.plot(reward_hist, label='Episode Reward')
    plt.plot(np.convolve(reward_hist, np.ones(20)/20, mode='valid'), label='20-Episode Avg')
    plt.title('MBPO Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid()
    plt.savefig('mbpo_training_2_1000steps.png')
    plt.show()
    print("⏹️ MBPO Training complete!")


# ----------------------------
#  Evaluation Loop
# ----------------------------
def evaluate(env_name="PandaReach-v3", episodes=10, max_steps=200):
    env = gym.make(env_name, render_mode="human", reward_type="dense")
    obs_space = env.observation_space
    flat_obs_dim = int(flatten_space(obs_space).shape[0])
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]

    agent = SACAgent(flat_obs_dim, act_dim, act_limit)
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
    train(episodes=1000, max_steps=50, render=True)
    #evaluate(episodes=100, max_steps=5)
