import random
import numpy as np

class Algorithm():
    def __init__(self, agent):
        self.M = agent.condition
        self._cond_idx = range(self.M)
        self.iteration_count = 0

        # Grab a few values from the agent.
        self.T = agent.T
        self.dU = agent.dU
        self.dX = agent.dX
        self.dO = agent.dO

        init_traj_distr = agent.traj_distr_lqg
        init_traj_distr['x0'] = agent.x0
        init_traj_distr['dX'] = agent.dX
        init_traj_distr['dU'] = agent.dU

        # IterationData objects for each condition.
        self.cur = [IterationData() for _ in range(self.M)]
        self.prev = [IterationData() for _ in range(self.M)]

        if self._hyperparams['fit_dynamics']:
            dynamics = self._hyperparams['dynamics']

        for m in range(self.M):
            self.cur[m].traj_info = TrajectoryInfo()
            if self._hyperparams['fit_dynamics']:
                self.cur[m].traj_info.dynamics = dynamics['type'](dynamics)
            init_traj_distr = extract_condition(
                self._hyperparams['init_traj_distr'], self._cond_idx[m]
            )
            self.cur[m].traj_distr = init_traj_distr['type'](init_traj_distr)

        self.traj_opt = hyperparams['traj_opt']['type'](
            hyperparams['traj_opt']
        )
        if type(hyperparams['cost']) == list:
            self.cost = [
                hyperparams['cost'][i]['type'](hyperparams['cost'][i])
                for i in range(self.M)
            ]
        else:
            self.cost = [
                hyperparams['cost']['type'](hyperparams['cost'])
                for _ in range(self.M)
            ]
        self.base_kl_step = self._hyperparams['kl_step']

    # Update dynamics model using all samples.
    def update_dynamics(self):

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
