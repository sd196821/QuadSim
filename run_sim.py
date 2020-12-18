from dynamics.quadrotor import Drone
import numpy as np

quad1 = Drone()
quad1.reset()
for t in range(100):
    quad1.step(np.zeros(quad1.dim_u))
    state = quad1.get_state()
    print(state)


