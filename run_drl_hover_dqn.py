import gym
# import gym_docking.envs
# from stable_baselines.common.policies import MlpPolicy
# from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import DQN
from stable_baselines.common.evaluation import evaluate_policy

env = gym.make('gym_docking:hovering-v0')

model = DQN('MlpPolicy', env, learning_rate=1e-3, prioritized_replay=True, verbose=1)
model.learn(total_timesteps=int(2e5))
model.save("dqn_hover")

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

obs = env.reset()
for i in range(1000):
    action, states = model.predict(obs)
    obs, reward, dones, info = env.step(action)


