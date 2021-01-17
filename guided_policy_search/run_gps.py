import copy

from QuadSim.dynamics.quadrotor import Drone
from QuadSim.guided_policy_search.utils.data_logger import DataLogger
from QuadSim.guided_policy_search.sample import sample
import policy as plc
import sample as smp
import numpy as np

class GPS():
    def __init__(self):
        self.Drone = Drone()
        self.dt = self.Drone.dt
        self.T = 1000
        self.substeps = 0
        self.dX = 13
        self.dU = 4
        self.dM = 0 # ref: mjc_reacher_images
        self.iter = 5  # outer loop total iteration
        self.condition = 1
        self.num_samples = 5  # number of samples per iter
        self.data_logger = DataLogger()
        self.data_dir = "./data"

        self.samples = [[] for _ in range(self.num_samples)]

        self.sample_dicts = {_:self.samples for _ in range(self.condition)}

        self.sample_on_policy = True
        self.policy = plc.policy_NN()
        self.noisy = True

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
        for itr in range(self.iter):
            for cond in range(self.condition):
                for i in range(self.num_samples):
                    self.take_traj_sample(cond, i)
                self.sample_dicts[cond] = self.samples

            self.take_iteration(itr)
            #
            # self.take_policy_sample()


    def take_traj_sample(self, cond, i_sample):
        if self.sample_on_policy is True:
            pol = self.policy
        else:
            pol = self.traj_distr_lqg[cond]

        if self.noisy:
            noise = sample.generate_noise(smooth=True, var=0.1, renorm=True)
        else:
            noise = np.zeros((self.T, self.dU))

        X = self.Drone.reset()
        O = X
        new_sample = smp(self.T, self.dX, self.dU)
        U = np.zeros([self.T, self.dU])
        new_sample.set(X=X, U=U)
        for t in range(self.T):
            X_t = new_sample.get_X(t=t)
            obs_t = new_sample.get_obs(t=t)
            U[t, :] = pol.act(X_t, obs_t, t, noise[t, :])
            if (t + 1) < self.T:
                for _ in range(self.substeps):
                    self.Drone.step(U[t, :])
                own_X = self.Drone.get_state()
                new_sample.set(own_X, U[t, :], t)

        self.samples[i_sample].append(new_sample)


    def take_iteration(self, itr):
        traj_sample = self.sample_dicts
        self.algotirhm.iteration(traj_sample)


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






