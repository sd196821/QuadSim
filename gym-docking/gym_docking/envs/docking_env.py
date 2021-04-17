import gym
from gym import error, spaces, utils
from gym.utils import seeding

from dynamics.quadrotor import Drone
import numpy as np
from utils.transform import quat2rot, rot2euler, euler2rot, rot2quat, rad2deg, deg2rad, euler2quat, quat2euler


class DockingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):

        self.chaser = Drone()
        self.target = Drone()

        self.state_chaser = np.zeros(13)
        self.state_target = np.zeros(13)
        self.rel_state = np.zeros(12)

        # self.steps_beyond_done = None

        #Chaser Initial State
        chaser_ini_pos = np.array([3, -50, 0.5])
        chaser_ini_att = euler2quat(np.array([deg2rad(0), deg2rad(0), 0]))
        chaser_ini_angular_rate = np.array([0, deg2rad(0), 0])
        self.chaser_ini_state = np.zeros(13)
        self.chaser_ini_state[0:3] = chaser_ini_pos
        self.chaser_ini_state[6:10] = chaser_ini_att
        self.chaser_ini_state[10:] = chaser_ini_angular_rate

        #Target Initial State
        target_ini_pos = np.array([10, -50, 5])
        target_ini_att = euler2quat(np.array([deg2rad(0), deg2rad(0), 0]))
        target_ini_angular_rate = np.array([0, deg2rad(0), 0])
        self.target_ini_state = np.zeros(13)
        self.target_ini_state[0:3] = target_ini_pos
        self.target_ini_state[6:10] = target_ini_att
        self.target_ini_state[10:] = target_ini_angular_rate

        #Target Final State
        target_pos_des = np.array([10, -50, 5])  # [x, y, z]
        target_att_des = euler2quat(np.array([deg2rad(0), deg2rad(0), deg2rad(0)]))
        self.target_state_des = np.zeros(13)
        self.target_state_des[0:3] = target_pos_des
        self.target_state_des[6:10] = target_att_des

        #Final Relative Error
        self.rel_pos_threshold = 1
        self.rel_vel_threshold = 0.1
        self.rel_att_threshold = np.array([deg2rad(0), deg2rad(0), deg2rad(0)])
        self.rel_att_rate_threshold = np.array([deg2rad(0), deg2rad(0), deg2rad(0)])

        #State Limitation
        chaser_low = self.chaser.state_lim_low
        chaser_high = self.chaser.state_lim_high

        target_low = self.target.state_lim_low
        target_high = self.target.state_lim_high

        # obs rel info: 12x1 [rel_pos, rel_vel, rel_rpy, rel_rpy_rate]
        obs_low = np.array([-20, -100, 0, -10, -10, -10, -np.pi, -np.pi/2, -np.pi, -10 * 2 * np.pi, -10 * 2 * np.pi, -10 * 2 * np.pi])
        obs_high = np.array([20, 0, 100, 10, 10, 10, np.pi, np.pi/2, np.pi, 10 * 2 * np.pi, 10 * 2 * np.pi, 10 * 2 * np.pi])

        # rel_low = np.array([60, 0, 100, 10, 10, 10, 1, 1, 1, 1, 10 * 2 * np.pi, 10 * 2 * np.pi, 10 * 2 * np.pi])

        self.action_space = spaces.Box(low=np.array([0, -10, -10, -10, 0, -10, -10, -10]), high=np.array([10, 10, 10, 10, 10, 10, 10, 10,]))
        self.observation_space = spaces.Box(low=obs_low, high=obs_high)

        self.seed()
        # self.reset()


    def step(self, action):
        reward = 0.0
        action_chaser = action[0:4]
        action_target = action[4:]
        self.state_target = self.target.step(action_target)
        self.state_chaser = self.chaser.step(action_chaser)

        # dock port relative state
        chaser_dp = self.chaser.get_dock_port_state()  # drone A
        target_dp = self.target.get_dock_port_state()  # drone B

        # rpy = quat2euler(self.state[6:10])
        # pos_error = self.state_des[0:3] - self.state[0:3]
        # vel_error = self.state_des[3:6] - self.state[3:6]
        # att_error = rot2euler(quat2rot(self.state_des[6:10])) - rpy
        # att_vel_error = self.state_des[10:] - self.state[10:]

        self.rel_state = state2rel(self.state_chaser, self.state_target, chaser_dp, target_dp)

        done_final = bool((np.linalg.norm(dp_rel_pos, 2) < 0.001)
                    and (np.linalg.norm(dp_rel_vel, 2) < 0.01)
                    and (np.abs(phi_A2B) < (deg2rad(10.0)))
                    and (np.abs(theta_A2B) < (deg2rad(10.0)))
                    and (np.abs(psi_A2B) < (deg2rad(10.0))))

        done_overlimit = bool(np.linalg.norm(dp_rel_pos, 2) > 10)

        done = bool(done_final or done_overlimit)

        if done_final:
            reward_docked = 1000
        else:
            reward_docked = 0

        #tbc
        if done_overlimit:
            reward = - 100
        elif done_final:
            reward = reward_docked
        else:
            reward = -(np.linalg.norm(dp_rel_pos, 2)) \
            - np.linalg.norm(dp_rel_vel, 2) \
            - np.linalg.norm(rel_euler_A2B, 2)

        return self.rel_state, reward, done, {}


    def reset(self):
        state_chaser = self.chaser.reset(self.chaser_ini_state)
        state_target = self.target.reset(self.target_ini_state)
        chaser_dp = self.chaser.get_dock_port_state()  # drone A
        target_dp = self.target.get_dock_port_state()  # drone B
        out = state2rel(state_chaser, state_target, chaser_dp, target_dp)
        return out


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
    dp_rel_pos = target_dp.pos - chaser_dp.pos
    dp_rel_vel = target_dp.vel - chaser_dp.vel

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



