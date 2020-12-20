import numpy as np
import scipy



class controller():
    """Controller Class"""

    def __init__(self, L):
        self.kp_roll = 10
        self.kp_pitch = 0
        self.kp_yaw = 0

        self.kd_roll = 20
        self.kd_pitch = 0
        self.kd_yaw = 0

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
        attitude_des = self.quat2euler(state_des[6:10])
        attitude_now = self.quat2euler(state_now[6:10])
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
                      (self.kp_pitch * e_angle[1] + self.kp_pitch * e_angular_rate[1]),
                      (self.kp_yaw * e_angle[2] + self.kp_yaw * e_angular_rate[2])])
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
    def quat2euler(quat):
        """
        Quaternion to Euler Angle
        :param quat:
        :return:
        """
        quat_w, quat_x, quat_y, quat_z = quat[0], quat[1], quat[2], quat[3]
        euler_x = np.arctan2(2*quat_w*quat_x + 2*quat_y*quat_z, quat_w*quat_w - quat_x*quat_x - quat_y*quat_y + quat_z*quat_z)
        euler_y = -np.arcsin(2*quat_x*quat_z - 2*quat_w*quat_y)
        euler_z = np.arctan2(2*quat_w*quat_z+2*quat_x*quat_y, quat_w*quat_w + quat_x*quat_x - quat_y*quat_y - quat_z*quat_z)
        return np.array([euler_x, euler_y, euler_z])
