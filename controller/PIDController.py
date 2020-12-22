import numpy as np
import scipy



class controller():
    """Controller Class"""

    def __init__(self, L, mass):
        self.kp_roll = 100 # 60; 10 ;15
        self.kp_pitch = 100
        self.kp_yaw = 70

        self.kd_roll = 60 # 50; 14.3; 21(5s)
        self.kd_pitch = 60 # 50
        self.kd_yaw = 13
        self.ff_yaw = 0

        self.kp_x = 0.3 # 0.1;0.2;0.3
        self.kp_y = 0.3
        self.kp_z = 20  # 10;

        self.kd_x = 0.9 # 0.4;0.6(12.5s) 0.7();0.9(10)
        self.kd_y = 0.9
        self.kd_z = 18  # 6;

        self.Kf = 0.8
        self.Km = 0.1

        self.g = 9.81
        self.mass = mass

        # Control allocation matrix:
        self.allocation_matrix = np.linalg.inv(np.array([[self.Kf, self.Kf, self.Kf, self.Kf],
                                                        [0, self.Kf * L, 0, -self.Kf * L],
                                                        [-self.Kf * L, 0, self.Kf * L, 0],
                                                        [self.Km, -self.Km, self.Km, -self.Km]]))

    def attitude_controller(self, state_des, state_now):
        """
        Attitude Controller
        :param state_des: Desired State[13]
        :param state_now: Current State[13]
        :return: M: output moment[3]
        """
        attitude_des = self.rot2euler(self.quat2rot(state_des[6:10]))
        attitude_now = self.rot2euler(self.quat2rot(state_now[6:10]))
        att_rate_des = state_des[10:]
        att_rate_now = state_now[10:]

        kp = np.array([self.kp_roll, self.kp_pitch, self.kp_yaw])
        kd = np.array([self.kd_roll, self.kd_pitch, self.kd_yaw])
        k = np.column_stack((kp, kd))
        # print(k)

        e_angle = attitude_des - attitude_now
        e_angular_rate = att_rate_des - att_rate_now
        # print(e_angle, e_angular_rate)
        # print(e_angle.shape, e_angular_rate.shape)
        e = np.vstack((e_angle, e_angular_rate))
        # print(e)
        # print(k, e)
        M = np.array([(self.kp_roll * e_angle[0] + self.kd_roll * e_angular_rate[0]),
                      (self.kp_pitch * e_angle[1] + self.kd_pitch * e_angular_rate[1]),
                      (self.ff_yaw + self.kp_yaw * e_angle[2] + self.kd_yaw * e_angular_rate[2])])
        # M = k @ e[:, 0] + k @ e[:, 1] + k @ e[:, 2]
        # print(M)
        return M

    def hover_controller(self, state_des, state_now):
        acc_des = np.zeros(3)
        e_pos = state_des[0:3] - state_now[0:3]
        e_vel = state_des[3:6] - state_now[3:6]
        acc_des[0] = self.kp_x * e_pos[0] + self.kd_x * e_vel[0]
        acc_des[1] = self.kp_y * e_pos[1] + self.kd_y * e_vel[1]
        acc_des[2] = self.kp_z * e_pos[2] + self.kd_z * e_vel[2]

        F = self.mass * self.g + self.mass * acc_des[2]

        att_des = self.rot2euler(self.quat2rot(state_des[6:10]))
        psi_des = att_des[2]

        phi_des = (acc_des[0] * np.sin(psi_des) - acc_des[1] * np.cos(psi_des)) / self.g
        theta_des = (acc_des[0] * np.cos(psi_des) + acc_des[1] * np.sin(psi_des)) / self.g

        roll_rate_des = 0
        pitch_rate_des = 0

        att_des[0] = phi_des
        att_des[1] = theta_des
        att_des[2] = psi_des

        state_des[6:10] = self.rot2quat(self.euler2rot(att_des))
        state_des[10] = roll_rate_des
        state_des[11] = pitch_rate_des

        return F, state_des

    def PID(self, state_des, state_now):
        F, state_des_c = self.hover_controller(state_des, state_now)
        M = self.attitude_controller(state_des_c, state_now)
        output = np.zeros(4)
        output[0] = F
        output[1:] = M
        return output

    def get_motor_output(self, u):
        """Calculate motor control command as angular velocity"""
        F = u[0]
        M = u[1:]
        rotor_omega = self.allocation_matrix @ u
        return rotor_omega

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
    def rot2euler(R):
        phi = np.arcsin(R[1, 2])
        psi = np.arctan2(-R[1, 0] / np.cos(phi), R[1, 1] / np.cos(phi))
        theta = np.arctan2(-R[0, 2] / np.cos(phi), R[2, 2] / np.cos(phi))
        return np.array([phi, theta, psi])

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