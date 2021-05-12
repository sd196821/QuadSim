# QuadSim

## Requirements:
### Basic
Python Version: 3.6
(Maybe need to set a virtual environment, or change system python to python3)  
Tkinter(for matplotlib): $ `sudo apt install python3-tk`  
Matplotlib  
Numpy  
Scipy

### For Reinforcement Learning
OpenAI gym  
Stable_baseline  
Tensorflow==1.15

## Install Enviroments for Gym
1. $`cd QuadSim`
2. $`pip install -e gym-docking`

## Running Method
### PID Controller
PID control: $`python run_sim_PID.py`
### Neural Hover Controller
1. Training PPO2 controller: $`python train_drl_hover_ppo2.py`
2. Run trained controller: $`python run_trained_drl_hover_ppo2.py`

# Environments
1.docking-v0: Target drone is hovering with PID controller  
2.docking-v1: Docking Env for imitation learning  
3.docking-v2: Target drone is flying in a constant velocity with PID controller. With random initial position. 


