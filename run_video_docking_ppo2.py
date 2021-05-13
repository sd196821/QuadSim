import gym
from stable_baselines.common.policies import MlpPolicy, ActorCriticPolicy, register_policy, nature_cnn, CnnPolicy
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


class CustomPolicy(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
        super(CustomPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=True)

        with tf.variable_scope("model", reuse=reuse):
            activ = tf.nn.relu

            extracted_features = nature_cnn(self.processed_obs, **kwargs)
            extracted_features = tf.layers.flatten(extracted_features)

            pi_h = extracted_features
            for i, layer_size in enumerate([128, 128]):
                pi_h = activ(tf.layers.dense(pi_h, layer_size, name='pi_fc' + str(i)))
            pi_latent = pi_h

            vf_h = extracted_features
            for i, layer_size in enumerate([128, 128]):
                vf_h = activ(tf.layers.dense(vf_h, layer_size, name='vf_fc' + str(i)))
            value_fn = tf.layers.dense(vf_h, 1, name='vf')
            vf_latent = vf_h

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self._value_fn = value_fn
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})

if __name__ == '__main__':
    saver = U.ConfigurationSaver(log_dir='./logs')
    logger.configure(folder=saver.data_dir)


    env = gym.make('gym_docking:docking-v3')
    env = VecNormalize(env, norm_obs=True, norm_reward=False,
                       clip_obs=255)

    checkpoint_callback = CheckpointCallback(save_freq=int(5e4), save_path='./logs/',
                                             name_prefix='rl_model_621_shaping_video_10M')


    model = PPO2(policy=MlpPolicy, env=env, verbose=1,
                 tensorboard_log="./ppo2_docking_tensorboard/",
                 lam=0.95,
                 gamma=0.99,  # lower 0.9 ~ 0.99
                 # n_steps=math.floor(cfg['env']['max_time'] / cfg['env']['ctl_dt']),
                 policy_kwargs=dict(
                     net_arch=[128, 128, 128, dict(pi=[128], vf=[128])], act_fun=tf.nn.relu),
                 n_steps=600,
                 ent_coef=0.00,
                 learning_rate=3e-4,
                 vf_coef=0.5,
                 max_grad_norm=0.5,
                 nminibatches=10,
                 noptepochs=10,
                 cliprange=0.2)

    # load trained model
    # model = PPO2.load("ppo2_docking_621_shaping_10M", env=env, tensorboard_log="./ppo2_docking_tensorboard/")

    model.learn(total_timesteps=int(10e6), callback=checkpoint_callback)

    # user defined ppo2
    # model.learn(total_timesteps=int(10e6), logger=logger, log_dir=saver.data_dir)

    model.save("ppo2_docking_307_shaping_video_10M")  # [b:reward_dock=10]

