import gym
from gym import error, spaces, utils
from gym.utils import seeding

from dynamics.quadrotor import Drone
import numpy as np
from utils.transform import quat2rot, rot2euler, euler2rot, rot2quat, rad2deg, deg2rad


class DockingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.drone_target = Drone()
        self.drone_active = Drone()

        self.state = np.zeros(13)

        self.steps_beyond_done = None

        self.pos_threshold = 1
        self.vel_threshold = 0.1

        ini_pos = np.array([0.0, 0.0, 5.0])
        ini_att = rot2quat(euler2rot(np.array([deg2rad(0), deg2rad(0), 0])))
        ini_angular_rate = np.array([0, deg2rad(0), 0])
        self.ini_state = np.zeros(13)
        self.ini_state[0:3] = ini_pos
        self.ini_state[6:10] = ini_att
        self.ini_state[10:] = ini_angular_rate

        pos_des = np.array([0.0, 0.0, 5.0])  # [x, y, z]
        att_des = rot2quat(euler2rot(np.array([deg2rad(0), deg2rad(0), deg2rad(0)])))
        self.state_des = np.zeros(13)
        self.state_des[0:3] = pos_des
        self.state_des[6:10] = att_des

        low = self.drone.state_lim_low
        high = self.drone.state_lim_high

        self.action_space = spaces.Box(low=np.array([0, -10, -10, -10]), high=np.array([10, 10, 10, 10]))
        self.observation_space = spaces.Box(low=low, high=high)

        self.seed()
        # self.reset()


    def step(self, action):


    def reset(self):


    def render(self):


    def close(self):




