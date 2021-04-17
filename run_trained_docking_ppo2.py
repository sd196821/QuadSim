import gym
# from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
# from stable_baselines.common.evaluation import evaluate_policy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils.transform import quat2rot, rot2euler, euler2rot, rot2quat, rad2deg, deg2rad, quat2euler

env = DummyVecEnv([lambda: gym.make("gym_docking:docking-v0")])
# gym.make('gym_docking:hovering-v0')
model = PPO2.load('ppo2_docking')

total_step = 1000
state = np.zeros((total_step, 12))
rpy = np.zeros((total_step, 3))
time = np.zeros(total_step)
u_all = np.zeros((total_step, 4))
done = False

obs = env.reset()
for t in range(total_step):
    action, states = model.predict(obs)
    obs, reward, done, info = env.step(action)

    state_now = obs.flatten()
    # print('u: ', action)
    # print('s: ', obs.flatten())
    u_all[t, :] = action.flatten()
    state[t, :] = state_now
    # rpy[t, :] = quat2eul(state_now[6:9]))
    # if done:
    #     break
    #    obs = env.reset()
    time[t] = t
    # print('pos:', state_now[0:3])
    # print('vel:', state_now[3:6])
    # print('euler:', state_now[6:9])
    # print('euler:', state_now[9:])
    # print(state_now)
    # print(state)

print(state.shape)
# time = np.linspace(0, total_step, total_step)
# Plot Results
plt.figure()
# # plt.subplot(2, 3, 1)
plt.plot(time, state[:, 0:3])
plt.legend(['rel x', 'rel y', 'rel z'])
# plt.xlabel("Time/s")
# plt.ylabel("Position/m")
# plt.title("Position")

# plt.figure()
# plt.subplot(2, 3, 2)
# plt.plot(time, state[:, 3:6])
# plt.legend(['rel vx', 'rel vy', 'rel vz'])
# plt.xlabel("Time/s")
# plt.ylabel("Velocity/m*s^-1")
# plt.title("Velocity")

# plt.figure()
# plt.subplot(2, 3, 3)
# plt.plot(time, rad2deg(state[:, 6:9]))
# plt.legend(['rel phi', 'rel theta', 'rel psi'])
# plt.xlabel("Time/s")
# plt.ylabel("Angle/deg")
# plt.title("Attitude")

# plt.figure()
# plt.subplot(2, 3, 4)
# plt.plot(time, rad2deg(state[:, 9:]))
# plt.legend(['rel p', 'rel q', 'rel r'])
# plt.xlabel("Time/s")
# plt.ylabel("Angular rate/deg*s^-1")
# plt.title("Angular Rates")

# plt.figure()
# plt.subplot(2, 3, 5)
# plt.plot(time, u_all[:, 1:])
# plt.legend(['Mx', 'My', 'Mz'])
# plt.xlabel("Time/s")
# plt.ylabel("Moment/Nm")
# plt.title("Control Moment")

# plt.figure()
# plt.subplot(2, 3, 6)
# plt.plot(time, u_all[:, 0])
# plt.xlabel("Time/s")
# plt.ylabel("Force/N")
# plt.title("Total Thrust")

# trajectory_fig = plt.figure()
# ax = Axes3D(trajectory_fig)
# ax.plot3D(state[:, 0], state[:, 1], state[:, 2])
# ax.set_xlabel("relative x")
# ax.set_ylabel("relative y")
# ax.set_zlabel("relative z")

plt.show()
