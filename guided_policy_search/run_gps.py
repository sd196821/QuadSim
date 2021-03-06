import copy

from QuadSim.dynamics.quadrotor import Drone
from QuadSim.guided_policy_search.utils.data_logger import DataLogger
from QuadSim.guided_policy_search.sample import sample
from QuadSim.guided_policy_search.cost import RAMP_CONSTANT,evallogl2term
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

        self.init_pol_wt = 0.1
        self.fit_dynamics = True
        self.dynamics = {'type': DynamicsLRPrior,
                        'regularization': 1e-6,
                        'prior':{
                            'type': DynamicsPriorGMM,
                            'max_clusters': 20,
                            'min_samples_per_cluster': 40,
                            'max_samples': 20
                            'strength': 1.0,
                             }
                         }

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

        self.traj_opt = {
            # Dual variable updates for non-PD Q-function.
            'del0': 1e-4,
            'eta_error_threshold': 1e16,
            'min_eta': 1e-8,
            'max_eta': 1e16,
            'cons_per_step': False,  # Whether or not to enforce separate KL constraints at each time step.
            'use_prev_distr': False,  # Whether or not to measure expected KL under the previous traj distr.
            'update_in_bwd_pass': True,  # Whether or not to update the TVLG controller during the bwd pass.
        }


        self.action_cost = {
            'wu': np.array([1, 1])
        }

        self.state_cost = {
            'wp': np.array([1, 1]), # State weights - must be set.
            'ramp_option': RAMP_CONSTANT,  # How target cost ramps over time.
            'l1': 0.0,
            'l2': 1.0,
            'alpha': 1e-2,
            'wp_final_multiplier': 1.0,  # Weight multiplier on final time step.
            'target_state': None,  # Target state - must be set.
            # 'wp': None,
        }

        self.cost = {
            'costs': [self.action_cost, self.state_cost],
            'weights': [1e-5, 1.0],
        }

        self.alg = {
            'inner_iterations': 1,  # Number of iterations.
            'min_eta': 1e-5,  # Minimum initial lagrange multiplier in DGD for
            # trajectory optimization.
            'kl_step': 0.2,
            'min_step_mult': 0.01,
            'max_step_mult': 10.0,
            'min_mult': 0.1,
            'max_mult': 5.0,
            # Trajectory settings.
            'initial_state_var': 1e-6,
            'init_traj_distr': None,  # A list of initial LinearGaussianPolicy
            # objects for each condition.
            # Trajectory optimization.
            'traj_opt': None,
            # Weight of maximum entropy term in trajectory optimization.
            'max_ent_traj': 0.0,
            # Dynamics hyperaparams.
            'dynamics': None,
            # Costs.
            'cost': None,  # A list of Cost objects for each condition.
            # Whether or not to sample with neural net policy (only for badmm/mdgps).
            'sample_on_policy': False,
            # Inidicates if the algorithm requires fitting of the dynamics.
            'fit_dynamics': True,
        }
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






