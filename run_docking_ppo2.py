import gym
# from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
# from stable_baselines.common.evaluation import evaluate_policy

#env = DummyVecEnv([lambda: gym.make("gym_docking:docking-v0")])
env = gym.make('gym_docking:hovering-v0')

model = PPO2('MlpPolicy', env, verbose=1, tensorboard_log="./ppo2_docking_tensorboard/",
             learning_rate=2.5e-2)
model.learn(total_timesteps=1000000)
model.save("ppo2_docking")
