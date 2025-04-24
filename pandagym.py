import gymnasium as gym
import panda_gym
import numpy as np
import time
import keyboard

env = gym.make("PandaReach-v3", render_mode="human")
obs, _ = env.reset()

delta = 0.2

while True:
    action = np.zeros(3)  # Correct shape for PandaReach-v3

    if keyboard.is_pressed("a"):
        action[1] += delta
    if keyboard.is_pressed("d"):
        action[1] -= delta
    if keyboard.is_pressed("s"):
        action[0] -= delta
    if keyboard.is_pressed("w"):
        action[0] += delta
    if keyboard.is_pressed("up"):
        action[2] += delta
    if keyboard.is_pressed("down"):
        action[2] -= delta

    print("Action:", action)  # Debug

    obs, reward, terminated, truncated, info = env.step(action)
    time.sleep(0.05)

    if terminated or truncated:
        obs, _ = env.reset()
