from stable_baselines import GAIL
from stable_baselines.gail import ExpertDataset

import gym
# from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines.common import set_global_seeds, make_vec_env
from stable_baselines.common.callbacks import CheckpointCallback
# from stable_baselines.common.evaluation import evaluate_policy
import tensorflow as tf



# env = DummyVecEnv([lambda: gym.make("gym_docking:docking-v0")])
# env = gym.make('gym_docking:docking-v1')


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

if __name__ == '__main__':
    env_id = 'gym_docking:docking-v1'
    num_cpu = 10  # Number of processes to use
    # Create the vectorized environment
    #env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    env = gym.make('gym_docking:docking-v1')

    # Stable Baselines provides you with make_vec_env() helper
    # which does exactly the previous steps for you:
    # env = make_vec_env(env, n_envs=num_cpu, seed=0)


    checkpoint_callback = CheckpointCallback(save_freq=int(5e4), save_path='./logs/',
                                             name_prefix='rl_model_621_gail_10M')

    dataset = ExpertDataset(expert_path='./expert_PID/expert_PID_new.npz',
                            traj_limitation=-1, batch_size=10)

    model = GAIL(policy='MlpPolicy', env=env, verbose=1,
                 tensorboard_log="./gail_docking_tensorboard/",
                 policy_kwargs=dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])], act_fun=tf.nn.relu),
                 expert_dataset=dataset
                 )

    # load trained model
    # model = PPO2.load("./ppo2_docking_621_random_pre.zip", env=env, tensorboard_log="./ppo2_docking_tensorboard/")

    model.learn(total_timesteps=int(10e6), callback=checkpoint_callback)
    model.save("gail_docking_621_10M")

