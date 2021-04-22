from stable_baselines.gail import generate_expert_traj

import gym
# from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
# from stable_baselines.common.evaluation import evaluate_policy
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils.transform import quat2rot, rot2euler, euler2rot, rot2quat, rad2deg, deg2rad, quat2euler, euler2quat
from controller.PIDController import controller

def info2array(info,tf):
    chaser_st = np.zeros((tf, 13))
    target_st = np.zeros((tf, 13))
    for i in range(tf):
        chaser_st[i, :] = info[i]['chaser']
        target_st[i, :] = info[i]['target']

    return chaser_st, target_st

#env = DummyVecEnv([lambda: gym.make("gym_docking:docking-v0")])
env =  gym.make('gym_docking:docking-v0')

total_step = 1000
rel_state = np.zeros((total_step, 12))
state = np.zeros((total_step, 12))
rpy = np.zeros((total_step, 3))
time = np.zeros(total_step)
u_all = np.zeros((total_step, 4))

done = False
tf = 0
info_lst = []

obs = env.reset()

control = controller(env.chaser.get_arm_length(), env.chaser.get_mass())
print(env.chaser.get_arm_length())

state_des = env.chaser_ini_state

kp=0.35
kd = 0

for t in range(total_step):
    # action, states = model.predict(obs, deterministic=True)
    obss = obs.flatten()
    if t != 0 :
        state_last = (info_lst[t-1])['chaser']
    else:
        state_last = env.chaser_ini_state
    # rel_state[t, 0:3] = obss[0:3]
    # rel_state[t, 3:6] = obss[3:6]
    # rel_state[t, 6:10] = euler2quat(obss[6:9])
    # rel_state[t, 10:] = obss[9:]
    # state_now = rel_state[t, :]
    des_vel = kp * (env.state_target[0:3] - env.state_chaser[0:3]) + kd * ( - env.state_chaser[3:6])
    state_des[3:6] = des_vel
    u = control.vel_controller(state_des, state_now, state_laskt)
    action = control.vel_controller(env.state_target, env.state_chaser)
    obs, reward, done, info = env.step(action)

    # state_now = obs.flatten()
    # print('u: ', action)
    # print('s: ', obs.flatten())
    u_all[t, :] = action.flatten()
    state[t, :] = obss
    # rpy[t, :] = quat2eul(state_now[6:9]))

    time[t] = t
    info_lst.append(info)
    # print('obs:', obs)
    # print('vel:', state_now[3:6])
    # print('euler:', state_now[6:9])
    # print('euler:', state_now[9:])
    # print(state_now)
    # print(state)
    tf = t
    # if done:
        # obs = env.reset()
    #    tf = t
    #    break

plt.figure()
plt.subplot(2, 3, 1)
plt.plot(time[0:tf], state[0:tf, 0:3])
plt.legend(['rel x', 'rel y', 'rel z'])
plt.xlabel("Time/s")
plt.ylabel("Position/m")
plt.title("Position")

# plt.figure()
plt.subplot(2, 3, 2)
plt.plot(time[0:tf], state[0:tf, 3:6])
plt.legend(['rel vx', 'rel vy', 'rel vz'])
plt.xlabel("Time/s")
plt.ylabel("Velocity/m*s^-1")
plt.title("Velocity")

# plt.figure()
plt.subplot(2, 3, 3)
plt.plot(time[0:tf], quat2euler(state[0:tf, 6:10]))
plt.legend(['rel phi', 'rel theta', 'rel psi'])
plt.xlabel("Time/s")
plt.ylabel("Angle/deg")
plt.title("Attitude")

# plt.figure()
plt.subplot(2, 3, 4)
plt.plot(time[0:tf], rad2deg(state[0:tf, 10:]))
plt.legend(['rel p', 'rel q', 'rel r'])
plt.xlabel("Time/s")
plt.ylabel("Angular rate/deg*s^-1")
plt.title("Angular Rates")

# plt.figure()
plt.subplot(2, 3, 5)
plt.plot(time[0:tf], u_all[0:tf, 1:])
plt.legend(['Mx', 'My', 'Mz'])
plt.xlabel("Time/s")
plt.ylabel("Moment/Nm")
plt.title("Control Moment")

# plt.figure()
plt.subplot(2, 3, 6)
plt.plot(time[0:tf], u_all[0:tf, 0])
plt.xlabel("Time/s")
plt.ylabel("Force/N")
plt.title("Total Thrust")

trajectory_fig = plt.figure()
ax = Axes3D(trajectory_fig)
ax.plot3D(state[0:tf, 0], state[0:tf, 1], state[0:tf, 2])
ax.set_xlabel("relative x")
ax.set_ylabel("relative y")
ax.set_zlabel("relative z")

#plt.figure()
chaser_st, target_st = info2array(info_lst,tf)
# chaser = info_states
# print(chaser_st)
#plt.plot(time[0:tf], (info_lst[0:tf])['chaser'])

chaser_traj_fig = plt.figure()
ax = Axes3D(chaser_traj_fig)
ax.plot3D(chaser_st[0:tf, 0], chaser_st[0:tf, 1], chaser_st[0:tf, 2])
ax.set_xlabel("chaser x")
ax.set_ylabel("chaser y")
ax.set_zlabel("chaser z")

target_traj_fig = plt.figure()
ax = Axes3D(target_traj_fig)
ax.plot3D(target_st[0:tf, 0], target_st[0:tf, 1], target_st[0:tf, 2])
ax.set_xlabel("target x")
ax.set_ylabel("target y")
ax.set_zlabel("target z")

plt.show()

