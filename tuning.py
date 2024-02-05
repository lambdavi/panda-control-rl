from typing import Any
from typing import Dict
import panda_gym
import gymnasium
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from stable_baselines3 import DDPG, SAC, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import torch
from argparse import ArgumentParser

parser = ArgumentParser(description="Tuning of models for Panda-Gym")

parser.add_argument(
    "--env_id",
    help="Id of the env",
    default="PandaReach-v3",
    required=False,
    choices=["PandaReach-v3", "PandaReachDense-v3", "PandaPickAndPlace-v3", "PandaPickAndPlaceDense-v3"],
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

N_TRIALS = 100
N_STARTUP_TRIALS = 20
N_EVALUATIONS = 2
N_TIMESTEPS = int(1e4)
EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)
N_EVAL_EPISODES = 3
ALGO = args.algo
ENV_ID = args.env_id

DEFAULT_HYPERPARAMS = {
    "policy": "MultiInputPolicy",
    "env": ENV_ID,
}


def sample_ddpg_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sampler for DDPG hyperparameters."""
    # buffer_size, learning_starts, batch_size, tau, 

    gamma = 1.0 - trial.suggest_float("gamma", 0.0001, 0.1, log=True)
    buffer_size = 10**trial.suggest_int("buffer_size", 2, 6)
    batch_size = 2**trial.suggest_int("batch_size", 5, 11)
    tau = trial.suggest_float("tau", 0.0005, 0.1, log=True)
    learning_starts = trial.suggest_int("learning_starts", 50, 200)
    learning_rate = trial.suggest_float("lr", 5e-4, 1, log=True)

    # Display true values.
    trial.set_user_attr("gamma_", gamma)
    trial.set_user_attr("buffer_size_", buffer_size)
    trial.set_user_attr("batch_size_", batch_size)
    

    return {
        "gamma": gamma,
        "buffer_size": buffer_size,
        "batch_size":batch_size,
        "tau":tau,
        "learning_starts": learning_starts,
        "learning_rate": learning_rate,
    }

def sample_sac_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sampler for SAC hyperparameters."""

    gamma = 1.0 - trial.suggest_float("gamma", 0.0001, 0.1, log=True)
    buffer_size = 10**trial.suggest_int("buffer_size", 2, 6)
    batch_size = 2**trial.suggest_int("batch_size", 5, 11)
    tau = trial.suggest_float("tau", 0.0005, 0.1, log=True)
    learning_starts = trial.suggest_int("learning_starts", 50, 200)
    learning_rate = trial.suggest_float("lr", 5e-4, 1, log=True)
    target_update_interval = trial.suggest_int("target_update_interval", 1, 10)


    # Display true values.
    trial.set_user_attr("gamma_", gamma)
    trial.set_user_attr("buffer_size_", buffer_size)
    trial.set_user_attr("batch_size_", batch_size)
    

    return {
        "gamma": gamma,
        "buffer_size": buffer_size,
        "batch_size":batch_size,
        "tau":tau,
        "learning_starts": learning_starts,
        "learning_rate": learning_rate,
        "target_update_interval": target_update_interval
    }

def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sampler for PPO hyperparameters."""
    # buffer_size, learning_starts, batch_size, tau, 
    n_steps = 2**trial.suggest_int("n_steps", 9, 12)
    n_epochs = trial.suggest_int("n_epochs", 5, 15)
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 1, log=True)
    gamma = 1.0 - trial.suggest_float("gamma", 0.0001, 0.1, log=True)
    batch_size = 2**trial.suggest_int("batch_size", 4, 9)
    learning_rate = trial.suggest_float("lr", 5e-4, 0.1, log=True)

    # Display true values.
    trial.set_user_attr("gamma_", gamma)
    trial.set_user_attr("batch_size_", batch_size)
    trial.set_user_attr("n_steps", n_steps)

    return {
        "n_steps": n_steps,
        "n_epochs": n_epochs,
        "gae_lambda": gae_lambda,
        "gamma": gamma,
        "batch_size":batch_size,
        "learning_rate": learning_rate,
    }


class TrialEvalCallback(EvalCallback):
    """Callback used for evaluating and reporting a trial."""

    def __init__(
        self,
        eval_env: gymnasium.Env,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 0,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if need.
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True


def objective(trial: optuna.Trial) -> float:
    kwargs = DEFAULT_HYPERPARAMS.copy()
    # Sample hyperparameters.
    # Create the RL model.
    if ALGO=="ddpg":
        kwargs.update(sample_ddpg_params(trial))
        model = DDPG(**kwargs)
    elif ALGO=="sac":
        kwargs.update(sample_sac_params(trial))
        model = SAC(**kwargs)
    elif ALGO=="ppo":
        kwargs.update(sample_ppo_params(trial))
        model = PPO(**kwargs)
    else:
        raise NotImplementedError
    # Create env used for evaluation.
    eval_env = Monitor(gymnasium.make(ENV_ID))
    # Create the callback that will periodically evaluate and report the performance.
    eval_callback = TrialEvalCallback(
        eval_env, trial, n_eval_episodes=N_EVAL_EPISODES, eval_freq=EVAL_FREQ, deterministic=True
    )

    nan_encountered = False
    try:
        model.learn(N_TIMESTEPS, callback=eval_callback)
    except AssertionError as e:
        # Sometimes, random hyperparams can generate NaN.
        print(e)
        nan_encountered = True
    finally:
        # Free memory.
        model.env.close()
        eval_env.close()

    # Tell the optimizer that the trial failed.
    if nan_encountered:
        return float("nan")

    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    return eval_callback.last_mean_reward


if __name__ == "__main__":
    # Set pytorch num threads to 1 for faster training.
    torch.set_num_threads(1)

    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    # Do not prune before 1/3 of the max budget is used.
    pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 3)

    study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")
    try:
        study.optimize(objective, n_trials=N_TRIALS, timeout=600)
    except KeyboardInterrupt:
        pass

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print("    {}: {}".format(key, value))