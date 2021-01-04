import gym
from gym import wrappers, logger
import numpy as np
from controller.PIDController import controller
from utils.transform import quat2rot, rot2euler, euler2rot, rot2quat, rad2deg, deg2rad


att_des = rot2quat(euler2rot(np.array([deg2rad(0), deg2rad(0), deg2rad(0)])))
pos_des = np.array([0, 0, 1])  # [x, y, z]
state_des = np.zeros(13)
state_des[0:3] = pos_des
state_des[6:10] = att_des
control = controller(0.086, 0.18)

env = gym.make('gym_docking:hovering-v0')

obs = env.reset()
for i in range(1000):
    env.seed(0)
    action = control.PID(state_des, obs)
    obs, reward, dones, info = env.step(action)
    # print('obs: ', obs)
    print('reward: ', reward)
    print('dones: ', dones)
