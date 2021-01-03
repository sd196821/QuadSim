import gym
from gym import wrappers, logger
import numpy as np

env = gym.make('gym_docking:hovering-v0')

obs = env.reset()
for i in range(1000):
    env.seed(0)
    obs, reward, dones, info = env.step(np.zeros(4))
    print(obs)
    print(reward)
