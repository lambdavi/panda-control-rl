import gymnasium as gym
import panda_gym
from time import sleep
from stable_baselines3 import DDPG
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

model = DDPG.load("PandaReach_DDPG_50000_steps")
env = gym.make('PandaReach-v3', render_mode="human")

observation, info = env.reset()

for _ in range(100):
    action, states_ = model.predict(observation=observation)
    observation, reward, terminated, truncated, info = env.step(action)
    sleep(0.05)
    if terminated or truncated:
        #print(f"Terminated with status: {'REACHED' if reward+1 > 0 else 'FAILED'}")
        observation, info = env.reset()
    
env.close()

# EVALUATING
print("EVALUATING")
eval_env = gym.make("PandaReach-v3", render_mode="human")

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