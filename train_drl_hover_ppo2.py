import gym
# from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines.common.callbacks import CheckpointCallback
import tensorflow as tf
from stable_baselines.common import set_global_seeds, make_vec_env

# env = DummyVecEnv([lambda: gym.make("gym_docking:hovering-v0")])
# gym.make('gym_docking:hovering-v0')

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
    env_id = 'gym_docking:hovering-v0'
    num_cpu = 4  # Number of processes to use
    # Create the vectorized environment
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

    # Stable Baselines provides you with make_vec_env() helper
    # which does exactly the previous steps for you:
    # env = make_vec_env(env, n_envs=num_cpu, seed=0)


    checkpoint_callback = CheckpointCallback(save_freq=int(5e4), save_path='./logs/',
                                             name_prefix='rl_model_hover_a_10M')

    model = PPO2(policy='MlpPolicy', env=env, verbose=1,
                 tensorboard_log="./ppo2_hovering_tensorboard/",
                 policy_kwargs=dict(
                     net_arch=[dict(pi=[128, 128], vf=[128, 128])], act_fun=tf.nn.relu),
                 lam=0.95,
                 gamma=0.99,  # lower 0.9 ~ 0.99
                 # n_steps=math.floor(cfg['env']['max_time'] / cfg['env']['ctl_dt']),
                 n_steps=200,
                 ent_coef=0.00,
                 learning_rate=6e-4,
                 vf_coef=0.5,
                 max_grad_norm=0.5,
                 nminibatches=1,
                 noptepochs=10,
                 cliprange=0.2)

    # load trained model
    # model = PPO2.load("./ppo2_docking_621_10M.zip", env=env, tensorboard_log="./ppo2_docking_tensorboard/")

    model.learn(total_timesteps=int(10e6), callback=checkpoint_callback)
    model.save("ppo2_hovering_621_a_10M")

