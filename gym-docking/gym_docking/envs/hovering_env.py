import gym
from gym import error, spaces, utils, logger
from gym.utils import seeding

from dynamics.quadrotor import Drone
import numpy as np
from utils.transform import quat2rot, rot2euler, euler2rot, rot2quat, rad2deg, deg2rad, euler2quat, quat2euler


class HoveringEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.drone = Drone()

        self.state = np.zeros(13)

        self.steps_beyond_done = None

        self.pos_threshold = 1
        self.vel_threshold = 0.1

        ini_pos = np.array([0.0, 0.0, 5.0]) + np.random.uniform(-1, 1, (3,))
        ini_att = euler2quat(np.array([deg2rad(0), deg2rad(0), 0]) + np.random.uniform(-0.2, 0.2, (3,)))
        ini_angular_rate = np.array([0, deg2rad(0), 0])
        self.ini_state = np.zeros(13)
        self.ini_state[0:3] = ini_pos
        self.ini_state[6:10] = ini_att
        self.ini_state[10:] = ini_angular_rate

        pos_des = np.array([0.0, 5.0, 20.0]) # [x, y, z]
        att_des = euler2quat(np.array([deg2rad(0), deg2rad(0), deg2rad(0)]))
        self.state_des = np.zeros(13)
        self.state_des[0:3] = pos_des
        self.state_des[6:10] = att_des

        low = self.drone.state_lim_low
        high = self.drone.state_lim_high

        self.action_space = spaces.Box(low=np.array([0.0, 0.0, 0.0, 0.0]), high=np.array([1.0, 1.0, 1.0, 1.0]))
        self.observation_space = spaces.Box(low=low, high=high)
        self.action_max = np.array([1.0, 1.0, 1.0, 1.0]) * self.drone.mass * self.drone.gravity

        self.seed()
        # self.reset()

    def step(self, action):
        reward = 0.0
        att_error = 0.0
        att_vel_error = 0.0
        u = self.drone.rotor2control @ (self.action_max * action[:])
        self.state = self.drone.step(u)
        # done1 = bool(-self.pos_threshold < np.linalg.norm(self.state[0:3] - self.state_des[0:3],
        # 2) < self.pos_threshold and -self.vel_threshold < np.linalg.norm(self.state[3:6] - self.state_des[3:6],
        # 2) < self.vel_threshold)\ or

        pos_error = self.state_des[0:3] - self.state[0:3]
        vel_error = self.state_des[3:6] - self.state[3:6]
        att_error = quat2euler(self.state_des[6:10]) - quat2euler(self.state[6:10])
        att_vel_error = self.state_des[10:] - self.state[10:]

        r_thre = 0.0
        if np.linalg.norm(pos_error, 2) < 0.1 and np.linalg.norm(vel_error, 2) < 0.1:
            r_thre = +1.0
        else:
            r_thre = 0.0

        done = bool((np.linalg.norm(self.state[0:3], 2) > 100) or (np.linalg.norm(self.state[3:6], 2) > 100))

        if not done:
            reward = r_thre + 0.1 - 0.01 * (np.linalg.norm(pos_error, 2)) \
                     - 0.001 *np.linalg.norm(vel_error, 2) \
                     - 0.01 * np.linalg.norm(att_error, 2) \
                     - 0.001 *np.linalg.norm(att_vel_error, 2)
        else:
            reward = -0.1
        # time = self.drone.get_time()
        return self.state, reward, done, {}

    def reset(self):
        out = self.drone.reset(self.ini_state)
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
