from stable_baselines.gail import generate_expert_traj
import gym
import warnings
from gym import spaces
import cv2  # pytype:disable=import-error

from controller.PIDController import controller
from stable_baselines.common import set_global_seeds
from stable_baselines.common.base_class import BaseRLModel
from stable_baselines.common.vec_env import VecEnv, VecFrameStack, SubprocVecEnv
from stable_baselines.common.callbacks import CheckpointCallback
from stable_baselines.common.base_class import _UnvecWrapper

from stable_baselines.gail import generate_expert_traj

import numpy as np

def info2array(info,tf):
    chaser_st = np.zeros((tf, 13))
    target_st = np.zeros((tf, 13))
    for i in range(tf):
        chaser_st[i, :] = info[i]['chaser']
        target_st[i, :] = info[i]['target']
    return chaser_st, target_st

env =  gym.make('gym_docking:docking-v0')

total_step = 1500
rel_state = np.zeros((total_step, 12))
state = np.zeros((total_step, 12))
rpy = np.zeros((total_step, 3))
time = np.zeros(total_step)
u_all = np.zeros((total_step, 4))

done = False
tf = 0
info_lst = []
rewards = []
obs = env.reset()

control = controller(env.chaser.get_arm_length(), env.chaser.get_mass())
state_des = env.chaser_ini_state

kp= 0.35
kd = 0

def generate_PID_expert_traj(save_path=None, env=None, n_timesteps=0,
                         n_episodes=1500, image_folder='recorded_images'):
    """
    Train expert controller (if needed) and record expert trajectories.

    .. note::

        only Box and Discrete spaces are supported for now.

    :param save_path: (str) Path without the extension where the expert dataset will be saved
        (ex: 'expert_cartpole' -> creates 'expert_cartpole.npz').
        If not specified, it will not save, and just return the generated expert trajectories.
        This parameter must be specified for image-based environments.
    :param env: (gym.Env) The environment, if not defined then it tries to use the model
        environment.
    :param n_timesteps: (int) Number of training timesteps
    :param n_episodes: (int) Number of trajectories (episodes) to record
    :param image_folder: (str) When using images, folder that will be used to record images.
    :return: (dict) the generated expert trajectories.
    """

    # Retrieve the environment using the RL model
    # if env is None and isinstance(model, BaseRLModel):
    #     env = model.get_env()

    assert env is not None, "You must set the env in the model or pass it to the function."

    is_vec_env = False
    if isinstance(env, VecEnv) and not isinstance(env, _UnvecWrapper):
        is_vec_env = True
        if env.num_envs > 1:
            warnings.warn("You are using multiple envs, only the data from the first one will be recorded.")

    # Sanity check
    assert (isinstance(env.observation_space, spaces.Box) or
            isinstance(env.observation_space, spaces.Discrete)), "Observation space type not supported"

    assert (isinstance(env.action_space, spaces.Box) or
            isinstance(env.action_space, spaces.Discrete)), "Action space type not supported"

    # Check if we need to record images
    obs_space = env.observation_space
    record_images = len(obs_space.shape) == 3 and obs_space.shape[-1] in [1, 3, 4] \
                    and obs_space.dtype == np.uint8
    if record_images and save_path is None:
        warnings.warn("Observations are images but no save path was specified, so will save in numpy archive; "
                      "this can lead to higher memory usage.")
        record_images = False

    if not record_images and len(obs_space.shape) == 3 and obs_space.dtype == np.uint8:
        warnings.warn("The observations looks like images (shape = {}) "
                      "but the number of channel > 4, so it will be saved in the numpy archive "
                      "which can lead to high memory usage".format(obs_space.shape))

    # if n_timesteps > 0 and isinstance(model, BaseRLModel):
    #     model.learn(n_timesteps)

    actions = []
    observations = []
    rewards = []
    episode_returns = np.zeros((n_episodes,))
    episode_starts = []

    ep_idx = 0
    obs = env.reset()
    episode_starts.append(True)
    reward_sum = 0.0
    idx = 0
    # state and mask for recurrent policies
    state, mask = None, None

    if is_vec_env:
        mask = [True for _ in range(env.num_envs)]

    while ep_idx < n_episodes:
        obs_ = obs[0] if is_vec_env else obs
        observations.append(obs_)

        # if isinstance(model, BaseRLModel):
        #     action, state = model.predict(obs, state=state, mask=mask)
        # else:
        if ep_idx != 0:
            state_last = (info_lst[ep_idx - 1])['chaser']
        else:
            state_last = env.chaser_ini_state

        des_vel = kp * (env.state_target[0:3] + np.array([-0.2, 0, 0]) - env.state_chaser[0:3]) + kd * (
            - env.state_chaser[3:6])
        if ep_idx != 0:
            state_des[3:6] = des_vel
        action = control.vel_controller(state_des, env.state_chaser, state_last)

        obs, reward, done, info = env.step(action)

        # Use only first env
        if is_vec_env:
            mask = [done[0] for _ in range(env.num_envs)]
            action = np.array([action[0]])
            reward = np.array([reward[0]])
            done = np.array([done[0]])

        actions.append(action)
        rewards.append(reward)
        info_lst.append(info)
        episode_starts.append(done)
        reward_sum += reward
        idx += 1

        if done:
            if not is_vec_env:
                obs = env.reset()
                # Reset the state in case of a recurrent policy
                state = None
            obs = env.reset()
            episode_returns[ep_idx] = reward_sum
            reward_sum = 0.0
            ep_idx += 1
        print("idx: ", idx, "  ep_idx: ", ep_idx)


    if isinstance(env.observation_space, spaces.Box) and not record_images:
        observations = np.concatenate(observations).reshape((-1,) + env.observation_space.shape)
    elif isinstance(env.observation_space, spaces.Discrete):
        observations = np.array(observations).reshape((-1, 1))
    elif record_images:
        observations = np.array(observations)

    if isinstance(env.action_space, spaces.Box):
        actions = np.concatenate(actions).reshape((-1,) + env.action_space.shape)
    elif isinstance(env.action_space, spaces.Discrete):
        actions = np.array(actions).reshape((-1, 1))

    rewards = np.array(rewards)
    episode_starts = np.array(episode_starts[:-1])

    assert len(observations) == len(actions)

    numpy_dict = {
        'actions': actions,
        'obs': observations,
        'rewards': rewards,
        'episode_returns': episode_returns,
        'episode_starts': episode_starts
    }  # type: Dict[str, np.ndarray]

    for key, val in numpy_dict.items():
        print(key, val.shape)

    if save_path is not None:
        np.savez(save_path, **numpy_dict)

    env.close()

    return numpy_dict

def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        # env = VecNormalize(env, norm_obs=True, norm_reward=True,
        #            clip_obs=10.)
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init

def dummy_expert(_obs):
    """
    Random agent. It samples actions randomly
    from the action space of the environment.

    :param _obs: (np.ndarray) Current observation
    :return: (np.ndarray) action taken by the expert
    """
    return env.action_space.sample()

if __name__ == '__main__':
    # env_id = 'gym_docking:docking-v0'
    # num_cpu = 4  # Number of processes to use
    # Create the vectorized environment
    # env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

    env = gym.make('gym_docking:docking-v0')
    # env.reset()
    # generate_PID_expert_traj('./expert_PID/expert_PID_new', env, n_episodes=10)
    generate_expert_traj(dummy_expert, './expert_PID/random_agent', env, n_episodes=1000)
