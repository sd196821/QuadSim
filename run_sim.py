from dynamics.quadrotor import Drone
from controller.PIDController import controller
from utils.transform import quat2rot, rot2euler, euler2rot, rot2quat, rad2deg, deg2rad
import numpy as np
import matplotlib.pyplot as plt

# print(rot2quat(euler2rot(np.array([0, 0, 0]))))
ini_att = rot2quat(euler2rot(np.array([deg2rad(0), deg2rad(0), 0])))
ini_angular_rate = np.array([0, deg2rad(0), 0])
ini_state = np.zeros(13)
ini_state[6:10] = ini_att
ini_state[10:] = ini_angular_rate

att_des = rot2quat(euler2rot(np.array([deg2rad(10), deg2rad(0), deg2rad(0)])))
pos_des = np.array([0, 0, 5])  # [x, y, z]
state_des = np.zeros(13)
state_des[0:3] = pos_des
state_des[6:10] = att_des

# Initial a drone and set its initial state
quad1 = Drone()
quad1.reset(ini_state)

control = controller(quad1.get_arm_length(), quad1.get_mass())

# Control Command
u = np.zeros(quad1.dim_u)
u[0] = quad1.get_mass() * 9.8
# u[3] = 0.2

total_step = 1000
state = np.zeros((total_step, 13))
rpy = np.zeros((total_step, 3))
time = np.zeros(total_step)
u_all = np.zeros((total_step, 4))

# Run simulation
for t in range(total_step):
    state_now = quad1.get_state()
    # u = control.PID(state_des, state_now)
    u[1:] = control.attitude_controller(state_des, state_now)
    u_all[t, :] = u
    state[t, :] = state_now
    rpy[t, :] = rot2euler(quat2rot(state_now[6:10]))
    time[t] = quad1.get_time()
    # print("time : ", time[t])
    quad1.step(u)

# Plot Results
plt.figure()
plt.plot(time, state[:, 0:3])
plt.legend(['x', 'y', 'z'])
plt.xlabel("Time/s")
plt.ylabel("Position/m")
plt.title("Position")

plt.figure()
plt.plot(time, state[:, 3:6])
plt.legend(['vx', 'vy', 'vz'])
plt.xlabel("Time/s")
plt.ylabel("Velocity/m*s^-1")
plt.title("Velocity")

plt.figure()
plt.plot(time, rad2deg(rpy))
plt.legend(['roll', 'pitch', 'yaw'])
plt.xlabel("Time/s")
plt.ylabel("Angle/deg")
plt.title("Attitude")

plt.figure()
plt.plot(time, rad2deg(state[:, 10:]))
plt.legend(['p', 'q', 'r'])
plt.xlabel("Time/s")
plt.ylabel("Angular rate/deg*s^-1")
plt.title("Angular Rates")

plt.figure()
plt.plot(time, u_all[:, 1:])
plt.legend(['Mx', 'My', 'Mz'])
plt.xlabel("Time/s")
plt.ylabel("Moment/Nm")
plt.title("Control Moment")

plt.figure()
plt.plot(time, u_all[:, 0])
plt.xlabel("Time/s")
plt.ylabel("Force/N")
plt.title("Total Thrust")

plt.show()



