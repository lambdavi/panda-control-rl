import gymnasium as gym
import panda_gym

from stable_baselines3 import DDPG
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback

# Save a checkpoint every 1000 steps
checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./logs/',
                                         name_prefix='rl_model')

env = gym.make("PandaReach-v3")

model = DDPG(
    "MultiInputPolicy",
    learning_rate=0.00001,
    env=env,        
    verbose=1,
)

model.learn(100_000, callback=checkpoint_callback)
env.close()
model.save("DDPG_Model_final")
