import numpy as np
from scipy.integrate import RK45


class Drone():
    """Quadrotor class"""

    def __init__(self):

        self.dt = 0.01
        self.t0 = 0
        self.t = self.t0
        # self.tf = 0.1

        self.gravity = 9.81
        self.mass = 0.18
        self.Inertia = np.array([[0.00025, 0, 0],
                                 [0, 0.000232, 0],
                                 [0, 0, 0.0003738]])
        self.arm_length = 0.086

        self.F_max = 4 * self.mass * self.gravity
        self.F_min = 0

        self.dim_state = 13  # x,y,z,vx,vy,vz,q1,q2,q3,q4,w1,w2,w3
        self.dim_u = 4

        self.state = np.zeros(shape=self.dim_state)
        # self.initial_state = np.zeros(self.dim_state)
        # self.initial_state[6] = 1.0
        self.initial_state = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])

        self.u = np.zeros(shape=self.dim_u)

        self.state_lim_low = np.array(
            [-100, -100, 0, -100, -100, -100, -100, -100, -100, -10 * 2 * np.pi, -10 * 2 * np.pi, -10 * 2 * np.pi])
        self.state_lim_high = np.array(
            [100, 100, 100, 100, 100, 100, 100, 100, 100, 10 * 2 * np.pi, 10 * 2 * np.pi, 10 * 2 * np.pi])

        self.integrator = None  # RK45(self.f, self.t0, self.state, self.tf)

        self.A = np.array([[0.25, 0, -0.5/self.arm_length],
                           [0.25, 0.5/self.arm_length, 0],
                           [0.25, 0, 0.5/self.arm_length],
                           [0.25, -0.5/self.arm_length, 0]])

        self.B = np.array([[1, 1, 1, 1],
                           [0, self.arm_length, 0, -self.arm_length],
                           [-self.arm_length, 0, self.arm_length, 0]])

        self.dock_port_inB = None
        self.dock_port_inB.pos = np.array([0.05, 0, 0])
        self.dock_port_inB.att = self.euler2rot(np.array([0, 0, 0]))

    def reset(self, reset_state=None):
        """
        Reset state, control and integrator
        """
        if reset_state is not None:
            self.initial_state = reset_state
        self.state = self.initial_state
        self.u = np.zeros(self.dim_u)
        self.integrator = RK45(self.f, self.t0, self.state, self.dt)
        return self.state

    def df(self, state, u):
        #  F, M1, M2, M3 = u
        F = u[0]
        M = self.Inertia @ u[1:]

        pos = state[0:3]
        vel = state[3:6]
        att_q = state[6:10]
        att_rate = state[10:]

        R_w2b = self.quat2rot(att_q)
        R_b2w = R_w2b.transpose()

        F_b = np.array([0, 0, F])
        acc = 1.0 / self.mass * (np.dot(R_b2w, F_b) - np.array([0, 0, self.mass * self.gravity]))

        K_quat = 2.0
        e_quat = 1.0 - np.sum(att_q ** 2)
        q_sk = np.array([[0, -att_rate[0], -att_rate[1], -att_rate[2]],
                         [att_rate[0], 0, -att_rate[1], att_rate[2]],
                         [att_rate[1], att_rate[2], 0, -att_rate[0]],
                         [att_rate[2], -att_rate[1], att_rate[0], 0]])

        q_dot = -0.5 * (q_sk @ att_q) + K_quat * e_quat * att_q

        att_acc = np.linalg.inv(self.Inertia) @ (M - np.cross(att_rate, (self.Inertia @ att_rate)))

        dstate = np.zeros(self.dim_state)
        dstate[0:3] = vel
        dstate[3:6] = acc
        dstate[6:10] = q_dot
        dstate[10:] = att_acc

        return dstate

    def f(self, t, df):
        """
        Repack df to match requirements of SciPy integrator
        :param t: Current Time
        :param df: Equation of Motion
        :return: Equation of Motion
        """
        t = self.t
        df = self.df(self.state, self.u)
        return df

    def step(self, u):
        """
        RK45 integrator for one step/dt
        :param u:Control Command--[F,Mx,My,Mz]
        :return: System State
        """
        while not (self.integrator.status == 'finished'):
            self.integrator.step()
        self.state = self.integrator.y
        self.u = self.u_limit(u)
        self.t = self.integrator.t
        self.integrator = RK45(self.f, self.integrator.t, self.state, self.integrator.t + self.dt)

        return self.state

    def u_limit(self, u):
        """
        Limit Force and Moment
        :param u:control
        :return: u_limited[F;M]
        """
        prop_thrust = self.A @ u[0:3]
        # prop_thrust_clamped = np.max(np.min(prop_thrust, (self.F_max / 4)), self.F_min)
        prop_thrust[prop_thrust > (self.F_max / 4)] = self.F_max / 4
        prop_thrust[prop_thrust < (self.F_min / 4)] = self.F_min / 4

        F = self.B[0, :] @ prop_thrust
        output = np.zeros(4)
        output[0] = F
        output[1:3] = self.B[1:, :] @ prop_thrust
        output[3] = u[3]
        return output

    def get_state(self):
        """
        Get system state
        :return: State of System
        """
        return self.state

    def get_time(self):
        """
        Get current time
        :return:Time
        """
        return self.t

    def get_arm_length(self):
        """
        Get arm length
        :return: Arm length
        """
        return self.arm_length

    def get_mass(self):
        return self.mass

    def get_dock_port_state(self):
        dock_port = None
        R_w2b = self.quat2rot(self.state[6:10])
        R_b2w = R_w2b.transpose()
        dock_port.pos = self.state[0:3] + R_b2w @ self.dock_port_inB.pos
        dock_port.quat = self.rot2quat(self.dock_port_inB.att @ R_w2b)
        w_sk = np.array([[0, -self.state[12], self.state[11]],
                         [self.state[12], 0, -self.state[10]],
                         [-self.state[11], self.state[10], 0]])
        dock_port.vel = self.state[3:6] + w_sk @ (R_b2w @ self.dock_port_inB.pos)
        dock_port.angular_rate = self.state[10:]
        return dock_port

    @staticmethod
    def quat2rot(quat):
        """
        Quaternion 2 Rotation Matrix Z-X-Y
        :param quat:Attitude Quaternion
        :return: Rotation Matrix
        """
        R = np.zeros((3, 3))
        quat_n = quat / np.linalg.norm(quat)
        qa_hat = np.zeros((3, 3))
        qa_hat[0, 1] = -quat_n[3]
        qa_hat[0, 2] = quat_n[2]
        qa_hat[1, 2] = -quat_n[1]
        qa_hat[1, 0] = quat_n[3]
        qa_hat[2, 0] = -quat_n[2]
        qa_hat[2, 1] = quat_n[1]

        R = np.eye(3) + 2 * qa_hat * qa_hat + 2 * quat[0] * qa_hat

        return R

    @staticmethod
    def euler2rot(angle):
        phi, theta, psi = angle
        R = np.array([[np.cos(psi) * np.cos(theta) - np.sin(phi) * np.sin(psi) * np.sin(theta),
                       np.cos(theta) * np.sin(psi) + np.cos(psi) * np.sin(phi) * np.sin(theta),
                       -np.cos(phi) * np.sin(theta)],
                      [-np.cos(phi) * np.sin(psi), np.cos(phi) * np.cos(psi), np.sin(phi)],
                      [np.cos(psi) * np.sin(theta) + np.cos(theta) * np.sin(phi) * np.sin(psi),
                       np.sin(psi) * np.sin(theta) - np.cos(psi) * np.cos(theta) * np.sin(phi),
                       np.cos(phi) * np.cos(theta)]], dtype=np.float32)
        return R

    @staticmethod
    def rot2quat(R):
        tr = R[0, 0] + R[1, 1] + R[2, 2]

        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2  # S = 4 * qw
            qw = 0.25 * S
            qx = (R[2, 1] - R[1, 2]) / S
            qy = (R[0, 2] - R[2, 0]) / S
            qz = (R[1, 0] - R[0, 1]) / S
        elif (R[0, 0] > R[1, 1]) & (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # S = 4 * qx
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # S = 4 * qy
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # S = 4 * qz
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S
        q = np.array([qw, qx, qy, qz], dtype=np.float32)
        q = q * np.sign(qw)
        return q
