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

        self.state = np.zeros(13)

        self.steps_beyond_done = None

        self.pos_threshold = 1
        self.vel_threshold = 0.1

        ini_pos = np.array([0, 0, 0])
        ini_att = rot2quat(euler2rot(np.array([deg2rad(0), deg2rad(0), 0])))
        ini_angular_rate = np.array([0, deg2rad(0), 0])
        self.ini_state = np.zeros(13)
        self.ini_state[0:3] = ini_pos
        self.ini_state[6:10] = ini_att
        self.ini_state[10:] = ini_angular_rate

        pos_des = np.array([0, 0, 1])  # [x, y, z]
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
        reward = 0.0
        self.state = self.drone.step(action)
        # done1 = bool(-self.pos_threshold < np.linalg.norm(self.state[0:3] - self.state_des[0:3],
        # 2) < self.pos_threshold and -self.vel_threshold < np.linalg.norm(self.state[3:6] - self.state_des[3:6],
        # 2) < self.vel_threshold)\ or

        rpy = rot2euler(quat2rot(self.state_des[6:10]))
        done = bool((np.linalg.norm(self.state[0:3], 2) < -10) or (np.linalg.norm(self.state[0:3], 2) > 10)
                    or (np.linalg.norm(self.state[3:6], 2) < -5) or (np.linalg.norm(self.state[3:6], 2) > 5)
                    or (np.abs(rpy[0]) > (deg2rad(89))) or (np.abs(rpy[1]) > (deg2rad(89))) or (np.abs(rpy[2]) > (deg2rad(179))))
        if not done:
            reward = - (np.linalg.norm(self.state_des[0:3] - self.state[0:3], 2)) + np.linalg.norm(
                rot2euler(quat2rot(self.state_des[6:10])) - rot2euler(quat2rot(self.state[6:10])), 2)
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
            reward = 10.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("Calling step though done!")
                self.steps_beyond_done += 1
                reward = 0.0
        # time = self.drone.get_time()
        return self.state, reward, done, {}

    def reset(self):
        out = self.drone.reset(self.ini_state)
        self.steps_beyond_done = None
        return out

    def render(self, mode='human'):
        return None

    def close(self):
        return None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

#    def get_time(self):
#        return self.drone.get_time()
