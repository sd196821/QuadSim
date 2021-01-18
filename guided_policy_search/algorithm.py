import random
import numpy as np
from QuadSim.guided_policy_search.utils.general_utils import extract_condition
from QuadSim.guided_policy_search.policy import policy_LQG
from QuadSim.guided_policy_search.traj_opt_lqr import TrajOptLQR
from QuadSim.guided_policy_search.dynamics_fit import DynamicsLRPrior


class Algorithm():
    def __init__(self, agent):
        self.M = agent.condition
        self._cond_idx = range(self.M)
        self.iteration_count = 0
        self.config = agent.alg

        # Grab a few values from the agent.
        self.T = agent.T
        self.dU = agent.dU
        self.dX = agent.dX
        self.dO = agent.dO

        init_traj_distr = agent.traj_distr_lqg
        init_traj_distr['x0'] = agent.x0
        init_traj_distr['dX'] = agent.dX
        init_traj_distr['dU'] = agent.dU

        self.sample_dicts = agent.sample_dicts

        # IterationData objects for each condition.
        self.cur = [IterationData() for _ in range(self.M)]
        self.prev = [IterationData() for _ in range(self.M)]

        if agent.fit_dynamics:
            self.dynamics = agent.dynamics

        for m in range(self.M):
            self.cur[m].traj_info = TrajectoryInfo()
            if agent.fit_dynamics:
                self.cur[m].traj_info.dynamics = DynamicsLRPrior(agent)
            init_traj_distr = extract_condition(
                agent.traj_distr_lqg, self._cond_idx[m]
            )
            self.cur[m].traj_distr = agent.traj_distr_lqg

        self.traj_opt = TrajOptLQR(agent)

        # TODO: After not done.
        self.cost = [
            hyperparams['cost']['type'](hyperparams['cost'])
            for _ in range(self.M)
        ]
        self.base_kl_step = self.config['kl_step']

    # Update dynamics model using all samples.
    def update_dynamics(self):
        for m in range(self.M):
            cur_data = self.sample_dicts[m]
            X = cur_data.get_X()
            U = cur_data.get_U()



    # KL Divergence step size.
    def update_step_size(self):
        for m in range(self.M):
            cur_data = self.cur[m].sample_list
            X = cur_data.get_X()
            U = cur_data.get_U()

            # Update prior and fit dynamics.
            self.cur[m].traj_info.dynamics.update_prior(cur_data)
            self.cur[m].traj_info.dynamics.fit(X, U)

            # Fit x0mu/x0sigma.
            x0 = X[:, 0, :]
            x0mu = np.mean(x0, axis=0)
            self.cur[m].traj_info.x0mu = x0mu
            self.cur[m].traj_info.x0sigma = np.diag(
                np.maximum(np.var(x0, axis=0),
                           self._hyperparams['initial_state_var'])
            )

            prior = self.cur[m].traj_info.dynamics.get_prior()
            if prior:
                mu0, Phi, priorm, n0 = prior.initial_state()
                N = len(cur_data)
                self.cur[m].traj_info.x0sigma += \
                    Phi + (N * priorm) / (N + priorm) * \
                    np.outer(x0mu - mu0, x0mu - mu0) / (N + n0)


class IterationData:
    """ Collection of iteration variables. """
    def __init__(self):
        self.sample_list = None,  # List of samples for the current iteration.
        self.traj_info = None,  # Current TrajectoryInfo object.
        self.pol_info = None,  # Current PolicyInfo object.
        self.traj_distr = None,  # Initial trajectory distribution.
        self.new_traj_distr = None, # Updated trajectory distribution.
        self.cs = None,  # Sample costs of the current iteration.
        self.step_mult = 1.0,  # KL step multiplier for the current iteration.
        self.eta = 1.0,  # Dual variable used in LQR backward pass.


class TrajectoryInfo:
    """ Collection of trajectory-related variables. """
    def __init__(self):
        self.dynamics = None,  # Dynamics object for the current iteration.
        self.x0mu = None,  # Mean for the initial state, used by the dynamics.
        self.x0sigma = None,  # Covariance for the initial state distribution.
        self.cc = None,  # Cost estimate constant term.
        self.cv = None,  # Cost estimate vector term.
        self.Cm = None,  # Cost estimate matrix term.
        self.last_kl_step = float('inf'),  # KL step of the previous iteration.


class PolicyInfo:
    """ Collection of policy-related variables. """
    def __init__(self, agent):
        T, dU, dX = agent.T, agent.dU, agent.dX
        self.lambda_k = np.zeros((T, dU)),  # Dual variables.
        self.lambda_K = np.zeros((T, dU, dX)),  # Dual variables.
        self.pol_wt = agent.init_pol_wt * np.ones(T),  # Policy weight.
        self.pol_mu = None,  # Mean of the current policy output.
        self.pol_sig = None,  # Covariance of the current policy output.
        self.pol_K = np.zeros((T, dU, dX)),  # Policy linearization.
        self.pol_k = np.zeros((T, dU)),  # Policy linearization.
        self.pol_S = np.zeros((T, dU, dU)),  # Policy linearization covariance.
        self.chol_pol_S = np.zeros((T, dU, dU)),  # Cholesky decomp of covar.
        self.prev_kl = None,  # Previous KL divergence.
        self.init_kl = None,  # The initial KL divergence, before the iteration.
        self.policy_samples = [],  # List of current policy samples.
        self.policy_prior = None,  # Current prior for policy linearization.


    def traj_distr(self):
        """ Create a trajectory distribution object from policy info. """
        T, dU, dX = self.pol_K.shape
        # Compute inverse policy covariances.
        inv_pol_S = np.empty_like(self.chol_pol_S)
        for t in range(T):
            inv_pol_S[t, :, :] = np.linalg.solve(
                self.chol_pol_S[t, :, :],
                np.linalg.solve(self.chol_pol_S[t, :, :].T, np.eye(dU))
            )
        return policy_LQG(self.pol_K, self.pol_k, self.pol_S,
                self.chol_pol_S, inv_pol_S)


def estimate_moments(X, mu, covar):
    """ Estimate the moments for a given linearized policy. """
    N, T, dX = X.shape
    dU = mu.shape[-1]
    if len(covar.shape) == 3:
        covar = np.tile(covar, [N, 1, 1, 1])
    Xmu = np.concatenate([X, mu], axis=2)
    ev = np.mean(Xmu, axis=0)
    em = np.zeros((N, T, dX+dU, dX+dU))
    pad1 = np.zeros((dX, dX+dU))
    pad2 = np.zeros((dU, dX))
    for n in range(N):
        for t in range(T):
            covar_pad = np.vstack([pad1, np.hstack([pad2, covar[n, t, :, :]])])
            em[n, t, :, :] = np.outer(Xmu[n, t, :], Xmu[n, t, :]) + covar_pad
    return ev, em


def gauss_fit_joint_prior(pts, mu0, Phi, m, n0, dwts, dX, dU, sig_reg):
    """ Perform Gaussian fit to data with a prior. """
    # Build weights matrix.
    D = np.diag(dwts)
    # Compute empirical mean and covariance.
    mun = np.sum((pts.T * dwts).T, axis=0)
    diff = pts - mun
    empsig = diff.T.dot(D).dot(diff)
    empsig = 0.5 * (empsig + empsig.T)
    # MAP estimate of joint distribution.
    N = dwts.shape[0]
    mu = mun
    sigma = (N * empsig + Phi + (N * m) / (N + m) *
             np.outer(mun - mu0, mun - mu0)) / (N + n0)
    sigma = 0.5 * (sigma + sigma.T)
    # Add sigma regularization.
    sigma += sig_reg
    # Conditioning to get dynamics.
    fd = np.linalg.solve(sigma[:dX, :dX], sigma[:dX, dX:dX+dU]).T
    fc = mu[dX:dX+dU] - fd.dot(mu[:dX])
    dynsig = sigma[dX:dX+dU, dX:dX+dU] - fd.dot(sigma[:dX, :dX]).dot(fd.T)
    dynsig = 0.5 * (dynsig + dynsig.T)
    return fd, fc, dynsig
