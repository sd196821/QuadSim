import gym
# from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
# from stable_baselines.common.evaluation import evaluate_policy
import tensorflow as tf

#env = DummyVecEnv([lambda: gym.make("gym_docking:docking-v0")])
env = gym.make('gym_docking:docking-v0')

model = PPO2(policy='MlpPolicy', env=env, verbose=1,
             tensorboard_log="./ppo2_docking_tensorboard/",
             policy_kwargs=dict(
                 net_arch=[dict(pi=[128, 128], vf=[128, 128])], act_fun=tf.nn.relu),
             lam=0.95,
             gamma=0.99,  # lower 0.9 ~ 0.99
             # n_steps=math.floor(cfg['env']['max_time'] / cfg['env']['ctl_dt']),
             n_steps=250,
             ent_coef=0.00,
             learning_rate=3e-4,
             vf_coef=0.5,
             max_grad_norm=0.5,
             nminibatches=1,
             noptepochs=10,
             cliprange=0.2)
model.learn(total_timesteps=250000)
model.save("ppo2_docking")
