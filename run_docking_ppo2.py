import gym
# from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines.common import set_global_seeds, make_vec_env
# from stable_baselines.common.evaluation import evaluate_policy
import tensorflow as tf

# env = DummyVecEnv([lambda: gym.make("gym_docking:docking-v0")])
# env = gym.make('gym_docking:docking-v0')

# class SaveOnBestTrainingRewardCallback(BaseCallback):
#     """
#     Callback for saving a model (the check is done every ``check_freq`` steps)
#     based on the training reward (in practice, we recommend using ``EvalCallback``).
#
#     :param check_freq: (int)
#     :param log_dir: (str) Path to the folder where the model will be saved.
#       It must contains the file created by the ``Monitor`` wrapper.
#     :param verbose: (int)
#     """
#     def __init__(self, check_freq: int, log_dir: str, verbose=1):
#         super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
#         self.check_freq = check_freq
#         self.log_dir = log_dir
#         self.save_path = os.path.join(log_dir, 'best_model')
#         self.best_mean_reward = -np.inf
#
#     def _init_callback(self) -> None:
#         # Create folder if needed
#         if self.save_path is not None:
#             os.makedirs(self.save_path, exist_ok=True)
#
#     def _on_step(self) -> bool:
#         if self.n_calls % self.check_freq == 0:
#
#           # Retrieve training reward
#           x, y = ts2xy(load_results(self.log_dir), 'timesteps')
#           if len(x) > 0:
#               # Mean training reward over the last 100 episodes
#               mean_reward = np.mean(y[-100:])
#               if self.verbose > 0:
#                 print("Num timesteps: {}".format(self.num_timesteps))
#                 print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))
#
#               # New best model, you could save the agent here
#               if mean_reward > self.best_mean_reward:
#                   self.best_mean_reward = mean_reward
#                   # Example for saving best model
#                   if self.verbose > 0:
#                     print("Saving new best model to {}".format(self.save_path))
#                   self.model.save(self.save_path)
#
#         return True


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
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init

if __name__ == '__main__':
    env_id = 'gym_docking:docking-v0'
    num_cpu = 4  # Number of processes to use
    # Create the vectorized environment
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

    # Stable Baselines provides you with make_vec_env() helper
    # which does exactly the previous steps for you:
    # env = make_vec_env(env_id, n_envs=num_cpu, seed=0)

    # model = ACKTR(MlpPolicy, env, verbose=1)

    model = PPO2(policy='MlpPolicy', env=env, verbose=1,
                 tensorboard_log="./ppo2_docking_tensorboard/",
                 policy_kwargs=dict(
                     net_arch=[dict(pi=[128, 128], vf=[128, 128])], act_fun=tf.nn.relu),
                 lam=0.95,
                 gamma=0.99,  # lower 0.9 ~ 0.99
                 # n_steps=math.floor(cfg['env']['max_time'] / cfg['env']['ctl_dt']),
                 n_steps=1000,
                 ent_coef=0.00,
                 learning_rate=3e-2,
                 vf_coef=0.5,
                 max_grad_norm=0.5,
                 nminibatches=1,
                 noptepochs=10,
                 cliprange=0.2)
    model.learn(total_timesteps=500000)
    model.save("ppo2_docking")


# model.learn(total_timesteps=250000)

