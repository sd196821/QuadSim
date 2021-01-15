import copy

from QuadSim.dynamics.quadrotor import Drone
from QuadSim.guided_policy_search.utils.data_logger import DataLogger
import policy as plc
import numpy as np

class GPS():
    def __init__(self):
        self.Drone = Drone()
        self.dt = self.Drone.dt
        self.T = 1000
        self.dX = 13
        self.dU = 4
        self.iter = 5  # outer loop total iteration
        self.condition = 1
        self.num_samples = 5  # number of samples per iter
        self.data_logger = DataLogger()
        self.data_dir = "./data"
        self.sample_dicts = {'traj'}

        self.sample_on_policy = True
        self.policy = plc.policy_NN()

        self.traj_distr_lqg = [{
            'x0': np.zeros(self.dX),
            'dX': self.dX,
            'dU': self.dU,
            'T': self.T,
            'dt': self.dt,
            'init_var': 1.0,
            'stiffness': 1.0,
            'stiffness_vel': 0.5,
            'final_weight': 1.0,
            # Parameters for guessing dynamics
            'init_acc': [],  # dU vector of accelerations, default zeros.
            'init_gains': [],  # dU vector of gains, default ones.
        }]
        self.traj_distr_pd = [{
            'x0': np.zeros(self.dX),
            'dX': self.dX,
            'dQ': self.dX,#
            'dU': self.dU,
            'T': self.T,
            'dt': self.dt,
            'init_var': 10.0,
            'pos_gains': 10.0,  # position gains
            'vel_gains_mult': 0.01,  # velocity gains multiplier on pos_gains
            'init_action_offset': None,

        }]

        # self.distr =

    def run(self):
        for itr in range(0, self.iter):
            for cond in range(self.condition):
                for i in range(self.num_samples):
                    self.take_traj_sample(itr, cond, i)

            # traj_sample_lists = [
            #     self.agent. get_samples(cond, -self._hyperparams['num_samples'])
            #     for cond in self._train_idx
            # ]
            #
            # self.take_iteration(itr, traj_sample_lists)
            #
            # self.take_policy_sample()


    def take_traj_sample(self, itr, cond, i):
        if self.sample_on_policy is True:
            pol = self.policy
        else:
            pol = self.traj_distr_lqg[cond]

        self.Drone.reset()
        new_sample =
        for t in range(self.T):



    def take_iteration(self):


    def take_policy_sample(self):

    def _log_data(self, itr, traj_sample_lists, pol_sample_lists=None):
        """
        Log data and algorithm, and update the GUI.
        Args:
            itr: Iteration number.
            traj_sample_lists: trajectory samples as SampleList object
            pol_sample_lists: policy samples as SampleList object
        Returns: None
        """
        self.data_logger.pickle(
            self.data_dir + ('traj_sample_itr_%02d.pkl' % itr),
            copy.copy(traj_sample_lists)
        )
        if pol_sample_lists:
            self.data_logger.pickle(
                self.data_dir + ('pol_sample_itr_%02d.pkl' % itr),
                copy.copy(pol_sample_lists)
            )






