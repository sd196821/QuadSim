import gym
# from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
# from stable_baselines.common.evaluation import evaluate_policy

env = DummyVecEnv([lambda: gym.make("gym_docking:hovering-v0")])
# gym.make('gym_docking:hovering-v0')

print()

model = PPO2('MlpPolicy', env, verbose=1)
model.learn(10000)
model.save("ppo2_hover")
