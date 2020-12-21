import numpy as np
import scipy



class controller():
    """Controller Class"""

    def __init__(self, L):
        self.kp_roll = 10 # 60; 10 ;15
        self.kp_pitch = 10
        self.kp_yaw = 70

        self.kd_roll = 14.3 # 50; 14.5; 21(5s)
        self.kd_pitch = 14.3 #50
        self.kd_yaw = 13
        self.ff_yaw = 0

        self.Kf = 0.8
        self.Km = 0.1

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