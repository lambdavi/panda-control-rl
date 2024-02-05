import gymnasium as gym
import panda_gym
from stable_baselines3 import DDPG, SAC, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from argparse import ArgumentParser
from os import path
parser = ArgumentParser(description="Tuning of models for Panda-Gym")

parser.add_argument(
    "--env_id",
    help="Id of the env",
    default="PandaReach-v3",
    required=True,
    choices=["PandaReach-v3", "PandaReachDense-v3", "PandaPickAndPlace-v3", "PandaPickAndPlaceDense-v3"],
    type=str,
)

parser.add_argument(
    "--algo",
    help="algorithm to solve the task",
    default="ddpg",
    required=True,
    choices=["ddpg", "sac", "dqn"],
    type=str,
)

parser.add_argument(
    "--path",
    help="path to the trained model",
    default=None,
    required=False,
    type=str,
)

args = parser.parse_args()

ALGO = args.algo
PATH = args.path
ENV_ID = args.env_id

if PATH is None:
    PATH = f"checkpoints/{ENV_ID[:-3]}_{ALGO.upper()}_final.zip"
    print(f"Path not specified. Trying from loading: {PATH}")

if not path.exists(PATH):
    raise FileNotFoundError
    
if ALGO=="ddpg":        
    model = DDPG.load(PATH)
elif ALGO == "sac":
    model = SAC.load(PATH)
elif ALGO == "ppo":
    model = PPO.load(PATH)
else:
    raise NotImplementedError

# EVALUATION START
print("EVALUATING")
eval_env = gym.make(ENV_ID, render_mode="human")

eval_env = Monitor(eval_env)
# Random Agent, before training
mean_reward, std_reward = evaluate_policy(
    model,
    eval_env,
    n_eval_episodes=3000,
    deterministic=True,
)

print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
eval_env.close()