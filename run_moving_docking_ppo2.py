import gym
from stable_baselines.common.policies import MlpPolicy, register_policy
from stable_baselines import PPO2
# from rl_baselines.ppo2.ppo2 import PPO2
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines.common import set_global_seeds, make_vec_env
from stable_baselines.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
# from stable_baselines.common.evaluation import evaluate_policy
import tensorflow as tf
from stable_baselines.common.schedules import LinearSchedule
from typing import Callable

from stable_baselines import logger
import rl_baselines.common.util as U

# env = DummyVecEnv([lambda: gym.make("gym_docking:docking-v0")])
# env = gym.make('gym_docking:docking-v0')

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

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value:
    :return: current learning rate depending on remaining progress
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)
        :return: (float)
        """
        if progress_remaining >= 0.96:
            rate = progress_remaining * initial_value
        else:
            rate = progress_remaining * float(0.00008)
        return rate

    return func

if __name__ == '__main__':
    saver = U.ConfigurationSaver(log_dir='./logs')
    logger.configure(folder=saver.data_dir)
    env_id = 'gym_docking:docking-v2'
    num_cpu = 10  # Number of processes to use
    # Create the vectorized environment
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

    # env = DummyVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    # [lambda: gym.make("gym_docking:docking-v0")])

    # Stable Baselines provides you with make_vec_env() helper
    # which does exactly the previous steps for you:
    # env = make_vec_env(env, n_envs=num_cpu, seed=0)
    eval_env = gym.make('gym_docking:docking-v2')
    eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/best_shaping_moving_b_10M_model',
                                 log_path='./logs/best_shaping_moving_b_10M_results', eval_freq=600)

    checkpoint_callback = CheckpointCallback(save_freq=int(5e4), save_path='./logs/',
                                             name_prefix='rl_model_621_shaping_moving_b_10M')

    # Create the callback list
    callback = CallbackList([checkpoint_callback, eval_callback])

    lr_sch = LinearSchedule(int(10e6), 1.0e-5, 2.5e-4)


    model = PPO2(policy=MlpPolicy, env=env, verbose=1,
                 tensorboard_log="./ppo2_docking_tensorboard/",
                 policy_kwargs=dict(
                     net_arch=[128, dict(pi=[128], vf=[128])], act_fun=tf.nn.relu),
                 lam=0.95,
                 gamma=0.99,  # lower 0.9 ~ 0.99
                 # n_steps=math.floor(cfg['env']['max_time'] / cfg['env']['ctl_dt']),
                 n_steps=600,
                 ent_coef=0.00,
                 learning_rate=3e-4,
                 # learning_rate=lr_sch.value,
                 # learning_rate=linear_schedule(3e-4),
                 vf_coef=0.5,
                 max_grad_norm=0.5,
                 nminibatches=10,
                 noptepochs=10,
                 cliprange=0.2)

    # load trained model
    # model = PPO2.load("ppo2_docking_621_shaping_10M", env=env, tensorboard_log="./ppo2_docking_tensorboard/")

    model.learn(total_timesteps=int(10e6), callback=callback)

    # user defined ppo2
    # model.learn(total_timesteps=int(10e6), logger=logger, log_dir=saver.data_dir)

    model.save("ppo2_docking_621_shaping_moving_10M")  # [b:reward_dock=10]
    # env.save("vec_normalize.pkl")


# model.learn(total_timesteps=250000)

