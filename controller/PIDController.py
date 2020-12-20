import numpy as np
import scipy



class controller():
    """Controller Class"""

    def __init__(self, L, state):
        self.kp_roll = 10
        self.kp_pitch =10
        self.kp_yaw = 20

        self.kd_roll = 1
        self.kd_pitch = 1
        self.kd_yaw = 1

        self.Kf = 1
        self.Km = 1

        # Control allocation matrix:
        self.allocation_matrix = np.linalg.inv(np.array([[self.Kf, self.Kf, self.Kf, self.Kf],
                                                        [0, self.Kf * L, 0 -self.Kf * L],
                                                        [-self.Kf * L, 0, self.Kf * L, 0],
                                                        [self.Km, -self.Km, self.Km, -self.Km]]))

    def attitude_controller(self):


    def get_motor_output(self, u):
        """Calculate motor control command as angular velocity"""
        F = u[0]
        M = u[1:]
        rotor_omega = self.allocation_matrix @ u
        return rotor_omega