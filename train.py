import gymnasium as gym
import panda_gym
import torch
import wandb
from argparse import ArgumentParser
from stable_baselines3 import DDPG, SAC, DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from wandb.integration.sb3 import WandbCallback

parser = ArgumentParser(description="Training of models for Panda-Gym")

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
    default=0.95,
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
    default=2048,
    required=False,
    type=int,
)
parser.add_argument(
    "--tau",
    help="tau value",
    default=0.05,
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
    default=150_000,
    required=False,
    type=int,
)
parser.add_argument(
    "--env_id",
    help="Id of the env",
    default="PandaReach-v3",
    required=False,
    choices=["PandaReach-v3", "PandaReachDense-v3", "PandaStack-v3", "PandaStackDense-v3", "PandaPickAndPlace-v3", "PandaPickAndPlaceDense-v3"],
    type=str,
)

parser.add_argument(
    "--algo",
    help="algorithm to solve the task",
    default="ddpg",
    required=False,
    choices=["ddpg", "sac", "dqn"],
    type=str,
)
args = parser.parse_args()


run = wandb.init(
    project="sb3",
    config=args,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # auto-upload the videos of agents playing the game
    # save_code=True,  # optional
)

## Parameters
lr = args.lr
gamma = args.gamma
buffer_size = args.buffer_size
bs = args.batch_size
tau = args.tau
learning_starts = args.learning_starts
steps=args.steps
env_id = args.env_id
algo = args.algo

# Save a checkpoint every 1000 steps
checkpoint_callback = CheckpointCallback(save_freq=5000, save_path='./checkpoints/',
                                         name_prefix=f'{env_id[:-3]}_{algo.upper()}')

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("ENV_ID: {}".format(env_id))
env = gym.make(env_id)
if algo=="ddpg":
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
        device=device
    )
elif algo=="sac":
    model = SAC(
        "MultiInputPolicy",
        learning_rate=lr,
        learning_starts=learning_starts,
        batch_size=bs,
        buffer_size=buffer_size,
        tau=tau,
        gamma=gamma,
        env=env,        
        verbose=1,
        tensorboard_log=f"runs/{run.id}",
        device=device,
    )
elif algo=="dqn":
    model = DQN(
        "MultiInputPolicy",
        learning_rate=lr,
        learning_starts=learning_starts,
        batch_size=bs,
        buffer_size=buffer_size,
        tau=tau,
        gamma=gamma,
        env=env,        
        verbose=1,
        device=device
    )
else:
    raise NotImplementedError
model.learn(steps, 
            callback=WandbCallback(
                    model_save_path=f"models/{run.id}",
                    verbose=2,
                ), 
            log_interval=50, 
            progress_bar=True)
env.close()
model.save(f"models/{env_id}_{algo.upper()}_final")
run.finish()
