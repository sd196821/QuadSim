from dynamics.quadrotor import Drone
from controller.PIDController import controller
from utils.transform import quat2euler, rad2deg, deg2rad
import numpy as np
import matplotlib.pyplot as plt

ini_state = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, deg2rad(5), 0, 0])

state_des = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0.0, 0.0, 0])

# Initial a drone and set its initial state
quad1 = Drone()
quad1.reset(ini_state)

att_control = controller(quad1.get_arm_length())

# Control Command
u = np.zeros(quad1.dim_u)
u[0] = 2 * 0.18
# u[3] = 0.2

total_step = 1000
state = np.zeros((total_step, 13))
rpy = np.zeros((total_step, 3))
time = np.zeros(total_step)
u_all = np.zeros((total_step, 4))


# Run sim
for t in range(total_step):
    state_now = quad1.get_state()
    u[1:] = att_control.attitude_controller(state_des, state_now)
    u_all[t, 1:] = u[1:]
    state[t, :] = state_now
    rpy[t, :] = quat2euler(state_now[6:10])
    time[t] = quad1.get_time()
    print("time : ", time[t])
    quad1.step(u)


plt.figure()
plt.plot(time, rad2deg(rpy))
plt.legend(['roll', 'pitch', 'yaw'])
plt.title("Attitude")

plt.figure()
plt.plot(time, rad2deg(state[:, 10:]))
plt.legend(['p', 'q', 'r'])
plt.title("Angular Rates")

plt.figure()
plt.plot(time, u_all[:, 1:])
plt.legend(['Mx', 'My', 'Mz'])
plt.title("Control Moment")

plt.show()



