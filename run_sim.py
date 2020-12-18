from dynamics.quadrotor import Drone
import numpy as np

quad1 = Drone()
quad1.reset()

u = np.zeros(quad1.dim_u)
u[0] = 2.0 * 9.8
u[3] = 0.2


for t in range(100):
    quad1.step(u)
    state = quad1.get_state()
    print(state[2], state[12])


