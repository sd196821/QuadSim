import numpy as np
import scipy


class Drone(object):
    """Quadrotor class"""

    def __init__(self):

        self.dt = 0.02

        self.gravity = 9.8
        self.mass = 2.0
        self.inertia = np.array([[10, 0, 0],
                                 [0, 10, 0],
                                 [0, 0, 10]])
        self.arm_length = 0.5

        self.dim_state = 13  # x,y,z,vx,vy,vz,q1,q2,q3,q4,w1,w2,w3
        self.dim_u = 4

        self.state = np.zeros(shape=self.dim_state)
        self.u = np.zeros(shape=self.dim_u)

        self.state_lim_low = np.array([-1000, -1000, 0, -100, -100, -100, -100, -100, -100, -10*2*np.pi, -10*2*np.pi, -10*2*np.pi])
        self.state_lim_high = np.array([1000, 1000, 1000, 100, 100, 100, 100, 100, 100, 10*2*np.pi, 10*2*np.pi, 10*2*np.pi])

    def reset(self):


    def df(self,state,u):


    def step(self,state,u):


