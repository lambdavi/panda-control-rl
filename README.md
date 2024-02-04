<div align="center">    
 
# Panda Control RL Suite   
![](https://github.com/lambdavi/panda-control-rl/blob/main/media/reach.gif?raw=true)
</div>

## Description   
The goal of this project is to handle every task in the **panda-gym** library. This repo can also be used as a starting point for further research without having to set-up the project from scratch.

- The simulation is based on PyBullet physics engine wrapped in Gymnasium.

- Environment created using: panda-gym.
- Baselines obtained with Stablebaselines3.

## How to run   
First, install dependencies (python==3.8 required)
```bash
# clone project   
git clone https://github.com/lambdavi/panda-control-rl.git

# install requirements   
cd panda-control-rl 
pip install -r requirements.txt
 ```   

# Available scripts
 ```bash
# run training script (example: training on PickAndPlace-v3 task)   
python train.py --env_id PandaPickAndPlace-v3 --algo ddpg

# run hyperparameters tuning (example: on PandaReach-v3 with SAC) 
python tuning.py --env_id PandaReach-v3 --algo sac

# eval your agent
python eval.py --env_id PandaReach-v3 --algo ddpg --path models/PandaReach_DDPG_50000_steps.zip

# just visualize the environment (random actions)
python visualize.py --env_id PandaReach-v3
```

#### Arguments for train:

- `--lr`: Learning rate
  - Type: float
  - Default: 0.001

- `--gamma`: Gamma value
  - Type: float
  - Default: 0.99

- `--buffer_size`: Buffer size
  - Type: int
  - Default: 1,000,000

- `--batch_size`: Batch size
  - Type: int
  - Default: 100

- `--tau`: Tau value
  - Type: float
  - Default: 0.005

- `--learning_starts`: When to start the learning
  - Type: int
  - Default: 100

- `--steps`: Number of steps
  - Type: int
  - Default: 50,000

- `--env_id`: Environment ID
  - Type: str
  - Default: "PandaReach-v3"
  - Choices: ["PandaReach-v3", "PandaReachDense-v3", "PandaPickAndPlace-v3", "PandaPickAndPlaceDense-v3"]

- `--algo`: Algorithm to solve the task
  - Type: str
  - Default: "ddpg"
  - Choices: ["ddpg", "sac", "dqn"]

### Citation   
```
@article{gallouedec2021pandagym,
    title        = {{panda-gym: Open-Source Goal-Conditioned Environments for Robotic Learning}},
    author       = {Gallou{\'e}dec, Quentin and Cazin, Nicolas and Dellandr{\'e}a, Emmanuel and Chen, Liming},
    year         = 2021,
    journal      = {4th Robot Learning Workshop: Self-Supervised and Lifelong Learning at NeurIPS},
    }
```   