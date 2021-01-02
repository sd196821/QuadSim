import gym
from gym import error, spaces, utils, logger
from gym.utils import seeding

from dynamics.quadrotor import Drone
import numpy as np
from utils.transform import quat2rot, rot2euler, euler2rot, rot2quat, rad2deg, deg2rad

class HoveringEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.drone = Drone()

        self.state = None

        self.steps_beyond_done = None

        self.pos_threshold = 10
        self.vel_threshold = 10

        ini_pos = np.array([0, 0, 0])
        ini_att = rot2quat(euler2rot(np.array([deg2rad(0), deg2rad(0), 0])))
        ini_angular_rate = np.array([0, deg2rad(0), 0])
        self.ini_state = np.zeros(13)
        self.ini_state[0:3] = ini_pos
        self.ini_state[6:10] = ini_att
        self.ini_state[10:] = ini_angular_rate

        low = self.drone.state_lim_low
        high = self.drone.state_lim_high

        self.act_space = spaces.Box(low=np.array([-20, -10, -10, -10]), high=np.array([20, 10, 10, 10]), shape=(4,))
        self.obs_space = space.Box(low=low, high=high, shape=(12,))

        self.seed()
        self.reset()

    def step(self, action):
        self.state = self.drone.step(action)
        done = bool(np.linalg.norm(self.state[0:3],2) < -self.pos_threshold
                    or np.linalg.norm(self.state[0:3],2) > self.pos_threshold
                    or np.linalg.norm(self.state[3:6], 2) < -self.vel_threshold
                    or np.linalg.norm(self.state[3:6], 2) > self.vel_threshold)
        if not done:
            reward = np.linalg.norm(self.state[0:3],2)
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("Calling step though done!")
                self.steps_beyond_done += 1
                reward = 0.0

        return self.state, reward, done, {}



    def reset(self):
        out = self.drone.reset(self.ini_state)
        self.steps_beyond_done = None
        return out

    def render(self):
        return None


    def close(self):
        return None

    def seed(self, seed=None):





