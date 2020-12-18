import numpy as np
import scipy


class Drone(object):
    """Quadrotor class"""

    def __init__(self):

        self.dt = 0.02

        self.gravity = 9.8
        self.mass = 2.0
        self.Inertia = np.array([[10, 0, 0],
                                 [0, 10, 0],
                                 [0, 0, 10]])
        self.arm_length = 0.5

        self.dim_state = 13  # x,y,z,vx,vy,vz,q1,q2,q3,q4,w1,w2,w3
        self.dim_u = 4

        self.state = np.zeros(shape=self.dim_state)
        self.u = np.zeros(shape=self.dim_u)

        self.state_lim_low = np.array([-1000, -1000, 0, -100, -100, -100, -100, -100, -100, -10*2*np.pi, -10*2*np.pi, -10*2*np.pi])
        self.state_lim_high = np.array([1000, 1000, 1000, 100, 100, 100, 100, 100, 100, 10*2*np.pi, 10*2*np.pi, 10*2*np.pi])

        self.initial_state = np.zeros(self.dim_state)

    def reset(self):
        """
        to be done
        """
        self.state = self.initial_state


    def df(self,state,u):

        #  F, M1, M2, M3 = u
        F = u[0]
        M = u[1:3]

        pos = state[0:2]
        vel = state[3:5]
        att_q = state[6:9]
        att_rate = state[10:12]

        R_w2b = self.quat2rot(att_q)
        R_b2w = R_w2b.transpose()

        F_b = np.array([0, 0, F])
        acc = 1.0 / self.mass * ( np.dot(R_b2w, F_b) - np.array([0, 0, self.mass*self.gravity]))

        K_quat = 2.0
        e_quat = 1.0 - np.sum(att_q**2)
        q_sk = np.array([[0, -att_rate[0], -att_rate[1], -att_rate[2]],
                                    [att_rate[0], 0, -att_rate[1], att_rate[2]],
                                    [att_rate[1], att_rate[2], 0, -att_rate[0]],
                                    [att_rate[2], -att_rate[1], att_rate[0], 0]])

        q_dot = -0.5 * q_sk * att_q + K_quat * e_quat * att_q

        att_acc = np.linalg.inv(self.Inertia) @ (M - np.outer(att_rate, self.Inertia@att_rate))

        dstate = np.zeros(self.dim_state)
        dstate[0:2] = vel
        dstate[3:5] = acc
        dstate[6:9] = q_dot
        dstate[10:12] = att_acc

        return dstate

    def step(self,state,u):


    @staticmethod
    def quat2rot(quat):
        """
        Quaternion 2 Rotation Matrix
        :param quat:attitude quaternion
        :return: Rotation Matrix
        """
        #  quat = np.zeros(4)
        R = np.zeros((3,3))

        quat_n = quat / np.linalg.norm(quat)
        qa_hat = np.zeros((2,1))
        qa_hat[0, 1] = -quat[3]
        qa_hat[0, 2] = quat[2]
        qa_hat[1, 2] = -quat[1]
        qa_hat[1, 0] = quat[3]
        qa_hat[2, 0] = -quat[2]
        qa_hat[2, 1] = quat[1]

        R = np.eye(3) + 2 * qa_hat * qa_hat + 2*quat[0]*qa_hat

        return R