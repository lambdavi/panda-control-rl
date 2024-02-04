<div align="center">    
 
# Panda Control RL Suite   
<!--
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/ICLR-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)  

ARXIV   
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)

![CI testing](https://github.com/PyTorchLightning/deep-learning-project-template/workflows/CI%20testing/badge.svg?branch=master&event=push)
-->

<!--  
Conference   
-->   
</div>

![](https://github.com/lambdavi/panda-control-rl/blob/main/media/reach.gif?raw=true)
## Description   
The goal of this project is to handle every task in the **panda-gym** library. This repo can also be used as a starting point for further research without having to set-up the project from scratch.

- The simulation is based on PyBullet physics engine wrapped in Gymnasium.

- Environment created using: panda-gym.
- Baselines obtained with Stablebaselines3.

## How to run   
First, install dependencies   
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

# run hyperparameters tuning (example: on PickAndPlace-v3 with SAC) 
python tuning.py --env_id PandaReach-v3 --algo sac

# Visualize agent in action 
python test.py --env_id PandaReach-v3 --algo ddpg --path models/PandaReach_DDPG_50000_steps.zip
```

### Citation   
```
@article{gallouedec2021pandagym,
title        = {{panda-gym: Open-Source Goal-Conditioned Environments for Robotic Learning}},
author       = {Gallou{\'e}dec, Quentin and Cazin, Nicolas and Dellandr{\'e}a, Emmanuel and Chen, Liming},
year         = 2021,
journal      = {4th Robot Learning Workshop: Self-Supervised and Lifelong Learning at NeurIPS},
}
```   