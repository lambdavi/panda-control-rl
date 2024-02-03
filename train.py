import gymnasium as gym
import panda_gym

from stable_baselines3 import DDPG
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback

# Save a checkpoint every 1000 steps
checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./logs/',
                                         name_prefix='rl_model')

MODE="train"
env = gym.make("PandaReach-v3")

if MODE == "test":
    observation, _ = env.reset()
    for _ in range(1000):
        current_position = observation["observation"][0:3]
        desired_position = observation["desired_goal"][0:3]
        action = 5.0 * (desired_position - current_position)
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            observation, info = env.reset()
    env.close()
else:
    model = DDPG(
        "MultiInputPolicy",
        learning_rate=0.00001,
        env=env,        
        verbose=1,
    )

    model.learn(30,callback=checkpoint_callback,)
    env.close()
    # EVALUATING
    print("EVALUATING")
    eval_env = gym.make("PandaReach-v3", render_mode="human")

    # Random Agent, before training
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=10,
        deterministic=True,
    )

    print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
    eval_env.close()
