from dynamics.quadrotor import Drone
from controller.PIDController import controller
from utils.transform import quat2rot, rot2euler, euler2rot, rot2quat, rad2deg, deg2rad, euler2quat, quat2euler
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from server.pub_server import pub_server as srv
import time
import cv2
from PIL import Image
import pyautogui as pag


# print(rot2quat(euler2rot(np.array([0, 0, 0]))))
ini_pos = np.array([8, -50, 5])
ini_att = euler2quat(np.array([deg2rad(0), deg2rad(0), 0]))
ini_angular_rate = np.array([0, deg2rad(0), 0])
ini_state = np.zeros(13)
ini_state[0:3] = ini_pos
ini_state[6:10] = ini_att
ini_state[10:] = ini_angular_rate

att_des = euler2quat(np.array([deg2rad(0), deg2rad(0), deg2rad(0)]))
pos_des = np.array([9.8, -50, 5])  # [x, y, z]
state_des = np.zeros(13)
state_des[0:3] = pos_des
state_des[6:10] = att_des

# Initial a drone and set its initial state
quad1 = Drone()
quad1.reset(ini_state)

chaser_pub_srv = srv(1)
target_pub_srv = srv(2)

control = controller(quad1.get_arm_length(), quad1.get_mass())

# Control Command
u = np.zeros(quad1.dim_u)
# u[0] = quad1.get_mass() * 9.81
# u[3] = 0.2

total_step = 600
state = np.zeros((total_step, 13))
rpy = np.zeros((total_step, 3))
sim_time = np.zeros(total_step)
u_all = np.zeros((total_step, 4))

# Run simulation
for t in range(total_step - 2):
    state_last = state[t - 1, :]
    state_now = quad1.get_state()

    # u[1:] = control.attitude_controller(state_des, state_now)
    des_vel = np.array([0.5, 0, 0])
    if t != 0:
        state_des[3:6] = des_vel
    u = control.vel_controller(state_des, state_now, state_last)

    u_all[t, :] = u
    state[t, :] = state_now
    rpy[t, :] = quat2euler(state_now[6:10])
    sim_time[t] = quad1.get_time()
    # print("time : ", time[t])
    # u[1]=0
    # u[2]=0
    # u[3]=0
    quad1.step(u)
    chaser_pub_srv.send_state(t, state_now)
    target_pub_srv.send_state(t, np.array([10, -50, 6.3, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]))
    # time.sleep(0.2)
    # try:
    # img = cv2.imread('/home/wang/Pictures/Training/' + str(t) + '.png')
    img = pag.screenshot('./alg_test/a'+str(t)+'.png', region=(0, 0, 1920, 1080))
    img.convert('L')
    img.resize((320, 240), Image.ANTIALIAS)
    img = np.array(img)
    # img.show()
    # aaa = ImageGrab.grab()
    # img.show()
    # cv2.imshow(img, cv2.COLOR_BGR2GRAY)
    # time.sleep(10)
    # print(img2.shape)
    # cv2.imshow('a', img)
    # cv2.waitKey(10)
    # except:
    #     time.sleep(1)
    #     img = cv2.imread('/home/wang/Pictures/Training/' + str(t) + '.png')
    #     img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # cv2.destroyWindow('step_' + str(t))

# Plot Results
plt.figure()
plt.subplot(2, 3, 1)
plt.plot(sim_time, state[:, 0:3])
plt.legend(['x', 'y', 'z'])
plt.xlabel("Time/s")
plt.ylabel("Position/m")
plt.title("Position")

# plt.figure()
plt.subplot(2, 3, 2)
plt.plot(sim_time, state[:, 3:6])
plt.legend(['vx', 'vy', 'vz'])
plt.xlabel("Time/s")
plt.ylabel("Velocity/m*s^-1")
plt.title("Velocity")

# plt.figure()
plt.subplot(2, 3, 3)
plt.plot(sim_time, rad2deg(rpy))
plt.legend(['roll', 'pitch', 'yaw'])
plt.xlabel("Time/s")
plt.ylabel("Angle/deg")
plt.title("Attitude")

# plt.figure()
plt.subplot(2, 3, 4)
plt.plot(sim_time, rad2deg(state[:, 10:]))
plt.legend(['p', 'q', 'r'])
plt.xlabel("Time/s")
plt.ylabel("Angular rate/deg*s^-1")
plt.title("Angular Rates")

# plt.figure()
plt.subplot(2, 3, 5)
plt.plot(sim_time, u_all[:, 1:])
plt.legend(['Mx', 'My', 'Mz'])
plt.xlabel("Time/s")
plt.ylabel("Moment/Nm")
plt.title("Control Moment")

# plt.figure()
plt.subplot(2, 3, 6)
plt.plot(sim_time, u_all[:, 0])
plt.xlabel("Time/s")
plt.ylabel("Force/N")
plt.title("Total Thrust")

trajectory_fig = plt.figure()
ax = Axes3D(trajectory_fig)
ax.plot3D(state[:, 0], state[:, 1], state[:, 2])
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

plt.show()