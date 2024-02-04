import gymnasium as gym
import panda_gym
from argparse import ArgumentParser
from stable_baselines3 import DDPG
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback

parser = ArgumentParser(description="Training of DDPG model for Panda-Gym")
parser.add_argument(
    "--lr",
    help="Learning rate",
    default=0.001,
    required=False,
    type=float,
)
parser.add_argument(
    "--gamma",
    help="Gamma value",
    default=0.99,
    required=False,
    type=float,
)
parser.add_argument(
    "--buffer_size",
    help="Buffer size",
    default=1_000_000,
    required=False,
    type=int,
)
parser.add_argument(
    "--batch_size",
    help="Batch size",
    default=100,
    required=False,
    type=int,
)
parser.add_argument(
    "--tau",
    help="tau value",
    default=0.005,
    required=False,
    type=float,
)
parser.add_argument(
    "--learning_starts",
    help="When to start the learning",
    default=100,
    required=False,
    type=int,
)
parser.add_argument(
    "--steps",
    help="#steps",
    default=50_000,
    required=False,
    type=int,
)
args = parser.parse_args()

## Parameters
lr = args.lr
gamma = args.gamma
buffer_size = args.buffer_size
bs = args.batch_size
tau = args.tau
learning_starts = args.learning_starts
steps=args.steps

# Save a checkpoint every 1000 steps
checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./logs/',
                                         name_prefix='rl_model')

env = gym.make("PandaReach-v3")

model = DDPG(
    "MultiInputPolicy",
    learning_rate=lr,
    learning_starts=learning_starts,
    batch_size=bs,
    buffer_size=buffer_size,
    tau=tau,
    gamma=gamma,
    env=env,        
    verbose=1,
)

model.learn(steps, callback=checkpoint_callback)
env.close()
model.save("DDPG_Model_final")
