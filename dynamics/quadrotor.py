import numpy as np
import scipy

class Drone():
    """Quadrotor class"""

    def __init__(self):
        self.mass = 2.0
        self.inertia = np.array([[10, 0, 0],
                                 [0, 10, 0],
                                 [0, 0, 10]])
        self.Nstate = 13 #x,y,z,vx,vy,vz,q1,q2,q3,q4,w1,w2,w3
        self.state = np.zeros()
