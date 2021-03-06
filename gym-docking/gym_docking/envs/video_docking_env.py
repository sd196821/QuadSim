import gym
from gym import error, spaces, utils
from gym.utils import seeding

from dynamics.quadrotor import Drone
import numpy as np
from utils.transform import quat2rot, rot2euler, euler2rot, rot2quat, rad2deg, deg2rad, euler2quat, quat2euler

from controller.PIDController import controller

from server.pub_server import pub_server as srv
from PIL import Image, ImageGrab

import time

class VideoDockingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):

        self.chaser = Drone()
        self.target = Drone()
        self.target_controller = controller(self.target.get_arm_length(), self.target.get_mass())

        self.state_chaser = np.zeros(13)
        self.state_target = np.zeros(13)
        self.rel_state = np.zeros(12)
        self.t = 0

        self.done = False
        self.reward = 0.0
        self.shaping = 0.0
        self.last_shaping = 0.0

        self.obs = np.zeros((240, 320, 3), dtype=np.uint8)
        self.chaser_pub_srv = srv(1)
        self.target_pub_srv = srv(2)

        # self.steps_beyond_done = None

        # Chaser Initial State
        chaser_ini_pos = np.array([8, -50, 5])  # + np.random.uniform(-0.5, 0.5, (3,))
        chaser_ini_vel = np.array([0, 0, 0])  # + np.random.uniform(-0.1, 0.1, (3,))
        chaser_ini_att = euler2quat(np.array([0.0, 0.0, 0.0]))  # + np.random.uniform(-0.2, 0.2, (3,)))
        chaser_ini_angular_rate = np.array([0.0, 0.0, 0.0])  # + np.random.uniform(-0.1, 0.1, (3,))
        self.chaser_dock_port = np.array([0.1, 0.0, 0.0])
        self.chaser_ini_state = np.zeros(13)
        self.chaser_ini_state[0:3] = chaser_ini_pos
        self.chaser_ini_state[3:6] = chaser_ini_vel
        self.chaser_ini_state[6:10] = chaser_ini_att
        self.chaser_ini_state[10:] = chaser_ini_angular_rate
        self.state_chaser = self.chaser.reset(self.chaser_ini_state, self.chaser_dock_port)

        # Target Initial State
        target_ini_pos = np.array([10, -50, 5])
        target_ini_vel = np.array([0.0, 0.0, 0.0])
        target_ini_att = euler2quat(np.array([0.0, 0.0, 0.0]))
        target_ini_angular_rate = np.array([0.0, 0.0, 0.0])
        self.target_dock_port = np.array([-0.1, 0, 0])
        self.target_ini_state = np.zeros(13)
        self.target_ini_state[0:3] = target_ini_pos
        self.target_ini_state[3:6] = target_ini_vel
        self.target_ini_state[6:10] = target_ini_att
        self.target_ini_state[10:] = target_ini_angular_rate
        self.state_target = self.target.reset(self.target_ini_state, self.target_dock_port)

        # Target Final State
        target_pos_des = np.array([10, -50, 5])  # [x, y, z]
        target_att_des = euler2quat(np.array([0.0, 0.0, 0.0]))
        self.target_state_des = np.zeros(13)
        self.target_state_des[0:3] = target_pos_des
        self.target_state_des[6:10] = target_att_des

        # Final Relative Error
        self.rel_pos_threshold = 1
        self.rel_vel_threshold = 0.1
        self.rel_att_threshold = np.array([deg2rad(0), deg2rad(0), deg2rad(0)])
        self.rel_att_rate_threshold = np.array([deg2rad(0), deg2rad(0), deg2rad(0)])

        # chaser_dp = self.chaser.get_dock_port_state()  # drone A
        # target_dp = self.target.get_dock_port_state()  # drone B
        self.rel_state = state2rel(self.state_chaser, self.state_target, self.chaser.get_dock_port_state(),
                                   self.target.get_dock_port_state())

        # State Limitation
        chaser_low = self.chaser.state_lim_low
        chaser_high = self.chaser.state_lim_high

        target_low = self.target.state_lim_low
        target_high = self.target.state_lim_high

        # obs rel info: 12x1 [rel_pos, rel_vel, rel_rpy, rel_rpy_rate]
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0, -1.0, -1.0]), high=np.array([1.0, 1.0, 1.0, 1.0]),
                                       dtype=np.float32)

        # Gray Image Observation
        self.observation_space = spaces.Box(low=0, high=255, shape=(240, 320, 3), dtype=np.uint8)

        # self.action_max = np.array([1.0, 1.0, 1.0, 1.0]) * self.chaser.mass * self.chaser.gravity
        self.action_mean = np.array([1.0, 1.0, 1.0, 1.0]) * self.chaser.mass * self.chaser.gravity / 2.0
        self.action_std = np.array([1.0, 1.0, 1.0, 1.0]) * self.chaser.mass * self.chaser.gravity / 2.0

        self.seed()
        # self.reset()

    def step(self, action):
        # last_reward = self.reward
        # reward = 0.0
        # shaping = 0.0
        self.t += 1

        old_state_target = self.state_target
        old_state_chaser = self.state_chaser
        old_rel_state = self.rel_state
        old_chaser_dp = self.chaser.get_dock_port_state()

        action_chaser = self.chaser.rotor2control @ (self.action_std * action[:] + self.action_mean)
        # action_chaser = self.chaser.rotor2control @ (self.action_max * action[:])

        action_target = self.target_controller.PID(self.target_state_des, self.state_target)
        self.state_target = self.target.step(action_target)
        self.state_chaser = self.chaser.step(action_chaser)

        self.chaser_pub_srv.send_state(int(self.t), self.state_chaser)
        self.target_pub_srv.send_state(int(self.t), self.state_target)

        img = ImageGrab.grab([0, 0, 1920, 1080])
        # img.convert('L')
        # time.sleep(0.1)
        resize_img = img.resize((320, 240), Image.ANTIALIAS)
        bbb = np.array(resize_img)
        self.obs = bbb

        # dock port relative state
        chaser_dp = self.chaser.get_dock_port_state()  # drone A
        target_dp = self.target.get_dock_port_state()  # drone B

        self.rel_state = state2rel(self.state_chaser, self.state_target, chaser_dp, target_dp)
        # done_final = False
        # done_overlimit = False
        flag_docking = bool((np.linalg.norm(self.rel_state[0:3], 2) < 0.1)
                            and (np.linalg.norm(self.rel_state[3:6], 2) < 0.1)
                            and (np.abs(self.rel_state[6]) < deg2rad(10))
                            and (np.abs(self.rel_state[7]) < deg2rad(10))
                            and (np.abs(self.rel_state[8]) < deg2rad(10)))

        done_overlimit = bool((np.linalg.norm(self.rel_state[0:3]) >= 3)
                              or self.state_chaser[2] <= 0.1)

        done_overtime = bool(self.t >= 600)

        self.done = bool(done_overlimit or done_overtime)

        reward_docked = 0
        if flag_docking:
            reward_docked = +1.0

        reward_action = np.linalg.norm(action[:], 2)

        self.shaping = - 10.0 * np.sqrt(np.sum(np.square(self.rel_state[0:3] / 3.0))) \
                       - 1.0 * np.sqrt(np.sum(np.square(self.rel_state[3:6]))) \
                       - 10.0 * np.sqrt(np.sum(np.square(self.rel_state[6:9] / np.pi))) \
                       - 1.0 * np.sqrt(np.sum(np.square(self.rel_state[9:]))) \
                       - 0.1 * reward_action + 1.0 * reward_docked

        self.reward = self.shaping - self.last_shaping
        self.last_shaping = self.shaping

        # reward += 0.1 * self.t

        info = {'chaser': self.state_chaser,
                'target': self.state_target,
                'flag_docking': flag_docking,
                'done_overlimit': done_overlimit}

        return self.obs, self.reward, self.done, info

    def reset(self):
        self.state_chaser = self.chaser.reset(self.chaser_ini_state, self.chaser_dock_port)
        self.state_target = self.target.reset(self.target_ini_state, self.target_dock_port)
        chaser_dp = self.chaser.get_dock_port_state()  # drone A
        target_dp = self.target.get_dock_port_state()  # drone B
        self.rel_state = state2rel(self.state_chaser, self.state_target, chaser_dp, target_dp)
        self.done = False
        self.obs = np.zeros((240, 320, 3), dtype=np.uint8)
        self.t = 0.0
        self.reward = 0.0
        self.shaping = 0.0
        self.last_shaping = 0.0
        return self.obs

    def render(self, mode='human'):
        return None

    def close(self):
        return None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


def state2rel(state_chaser, state_target, chaser_dp, target_dp):
    R_I2B = quat2rot(state_target[6:10])
    R_I2A = quat2rot(state_chaser[6:10])
    R_A2I = R_I2A.transpose()

    # relative pos & vel error
    dp_rel_pos = target_dp['pos'] - chaser_dp['pos']
    dp_rel_vel = target_dp['vel'] - chaser_dp['vel']

    # relative attitude & angular velocity error
    R_A2B = R_I2B @ R_A2I

    rel_euler_A2B = rot2euler(R_A2B)
    phi_A2B = rel_euler_A2B[0]
    theta_A2B = rel_euler_A2B[1]
    psi_A2B = rel_euler_A2B[2]

    omega_B = state_target[10:]
    omega_A = state_chaser[10:]
    rel_A2B = omega_B - omega_A
    rel_A2B_inB = R_I2B @ omega_B - R_A2B @ R_I2A @ omega_A
    p_A2B_inB = rel_A2B_inB[0]
    q_A2B_inB = rel_A2B_inB[1]
    r_A2B_inB = rel_A2B_inB[2]

    # relative angular rate
    dphi_A2B = p_A2B_inB * np.cos(theta_A2B) + r_A2B_inB * np.sin(theta_A2B)
    dtheta_A2B = q_A2B_inB - np.tan(phi_A2B) * (r_A2B_inB * np.cos(theta_A2B) - p_A2B_inB * np.sin(theta_A2B))
    dpsi_A2B = (r_A2B_inB * np.cos(theta_A2B) - p_A2B_inB * np.sin(theta_A2B)) / np.cos(phi_A2B)

    rel_state = np.zeros(12)
    rel_state[0:3] = dp_rel_pos
    rel_state[3:6] = dp_rel_vel
    rel_state[6:9] = rel_euler_A2B
    rel_state[9] = dphi_A2B
    rel_state[10] = dtheta_A2B
    rel_state[11] = dpsi_A2B

    return rel_state
