from dynamics.quadrotor import Drone
import numpy as np
import matplotlib.pyplot as plt

# Initial a drone and set its initial state
quad1 = Drone()
quad1.reset()

# Control Command
u = np.zeros(quad1.dim_u)
u[0] = 0 * 9.8
u[3] = 0.2

state = np.zeros((100, 13))
time = np.zeros(100)

# Run sim
for t in range(100):
    quad1.step(u)
    state[t, :] = quad1.get_state()
    time[t] = quad1.get_time()

plt.plot(time, state[:, 0:3])
plt.show()



