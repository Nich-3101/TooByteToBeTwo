import gymnasium as gym
import panda_gym
import time

env = gym.make('PandaPickAndPlace-v3', render_mode="human")

observation, info = env.reset()

for _ in range(1000):
    time.sleep(0.1)  # Sleep to slow down the simulation for better visualization
    action = env.action_space.sample() # random action
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()