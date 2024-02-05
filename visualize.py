import gymnasium as gym
import panda_gym
from time import sleep
from argparse import ArgumentParser

parser = ArgumentParser(description="Visualize environments")
parser.add_argument(
    "--env_id",
    help="Id of the env",
    default="PandaReach-v3",
    required=False,
    choices=["PandaReach-v3", "PandaReachDense-v3", "PandaPickAndPlace-v3", "PandaPickAndPlaceDense-v3"],
    type=str,
)
args = parser.parse_args()

env = gym.make(args.env_id, render_mode="human")

observation, info = env.reset()

for _ in range(100):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    sleep(0.05)
    if terminated or truncated:
        observation, info = env.reset()
    
env.close()