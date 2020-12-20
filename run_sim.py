from dynamics.quadrotor import Drone
from controller.PIDController import controller
import numpy as np
import matplotlib.pyplot as plt

ini_state = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0.001, 0])

state_des = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])

# Initial a drone and set its initial state
quad1 = Drone()
quad1.reset(ini_state)

att_control = controller(quad1.get_arm_length())

# Control Command
u = np.zeros(quad1.dim_u)
u[0] = 2 * 9.8
# u[3] = 0.2

total_step = 100
state = np.zeros((total_step, 13))
time = np.zeros(total_step)
u_all = np.zeros((total_step, 4))



# Run sim
for t in range(total_step):
    state_now = quad1.get_state()
    u[1:] = att_control.attitude_controller(state_des, state_now)
    u_all[t, 1:] = u[1:]
    state[t, :] = state_now
    time[t] = quad1.get_time()
    print("time : ", time[t])
    quad1.step(u)

plt.figure()
plt.plot(time, state[:, 10:])
plt.legend(['p', 'q', 'r'])

# plt.figure()
# plt.plot(time, u_all[:, 1:], label=('Mx', 'My', 'Mz'))
# plt.legend()
plt.show()



