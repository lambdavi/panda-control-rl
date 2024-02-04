import gymnasium as gym
import panda_gym
from time import sleep

env = gym.make('PandaReach-v3', render_mode="human")

observation, info = env.reset()

for _ in range(100):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    sleep(0.05)
    if terminated or truncated:
        observation, info = env.reset()
    
env.close()