import numpy as np

RAMP_CONSTANT = 1
RAMP_LINEAR = 2
RAMP_QUADRATIC = 3
RAMP_FINAL_ONLY = 4

class CostAction():
    def __init__(self, agent):
        self.config = agent.action_cost

    def eval(self, sample):
        """
        Evaluate cost function and derivatives on a sample.
        Args:
            sample: A single sample
        """
        sample_u = sample.get_U()
        T = sample.T
        Du = sample.dU
        Dx = sample.dX
        l = 0.5 * np.sum(self.config['wu'] * (sample_u ** 2), axis=1)
        lu = self.config['wu'] * sample_u
        lx = np.zeros((T, Dx))
        luu = np.tile(np.diag(self.config['wu']), [T, 1, 1])
        lxx = np.zeros((T, Dx, Dx))
        lux = np.zeros((T, Du, Dx))
        return l, lx, lu, lxx, luu, lux

class CostState():
    def __init__(self, agent):
        self.config = agent.state_cost

    def eval(self, sample):
        """
        Evaluate cost function and derivatives on a sample.
        Args:
            sample:  A single sample
        """
        T = sample.T
        Du = sample.dU
        Dx = sample.dX

        final_l = np.zeros(T)
        final_lu = np.zeros((T, Du))
        final_lx = np.zeros((T, Dx))
        final_luu = np.zeros((T, Du, Du))
        final_lxx = np.zeros((T, Dx, Dx))
        final_lux = np.zeros((T, Du, Dx))

        wp = self.config['wp']
        tgt = self.config['target_state']
        x = sample.get_X()
        _, dim_sensor = x.shape

        wpm = get_ramp_multiplier(
            self.config['ramp_option'], T,
            wp_final_multiplier=self.config['wp_final_multiplier']
        )
        wp *= np.expand_dims(wpm, axis=-1)
        # Compute state penalty.
        dist = x - tgt

        # Evaluate penalty term.
        l, ls, lss = evall1l2term(
            wp, dist, np.tile(np.eye(dim_sensor), [T, 1, 1]),
            np.zeros((T, dim_sensor, dim_sensor, dim_sensor)),
            self.config['l1'], self.config['l2'],
            self.config['alpha']
        )

        final_l += l

        # sample.agent.pack_data_x(final_lx, ls, data_types=[data_type])
        # sample.agent.pack_data_x(final_lxx, lss,
        #                              data_types=[data_type, data_type])
        final_lx = ls
        final_lxx = lss
        return final_l, final_lx, final_lu, final_lxx, final_luu, final_lux

class CostSum():
    def __init__(self, agent):
        self.config = agent.cost

        self._costs = []
        self._weights = self.config['weights']

    def eval(self, sample):
        """
        Evaluate cost function and derivatives.
        Args:
            sample:  A single sample
        """
        l, lx, lu, lxx, luu, lux = self._costs[0].eval(sample)

        # Compute weighted sum of each cost value and derivatives.
        weight = self._weights[0]
        l = l * weight
        lx = lx * weight
        lu = lu * weight
        lxx = lxx * weight
        luu = luu * weight
        lux = lux * weight
        for i in range(1, len(self._costs)):
            pl, plx, plu, plxx, pluu, plux = self._costs[i].eval(sample)
            weight = self._weights[i]
            l = l + pl * weight
            lx = lx + plx * weight
            lu = lu + plu * weight
            lxx = lxx + plxx * weight
            luu = luu + pluu * weight
            lux = lux + plux * weight
        return l, lx, lu, lxx, luu, lux




# class CostFK():
#     """
#     Forward kinematics cost function. Used for costs involving the end
#     effector position.
#     """
#     def __init__(self, agent):
#         self.config = agent.Cost_FK
#
#     def eval(self, sample):
#         """
#         Evaluate forward kinematics (end-effector penalties) cost.
#         Temporary note: This implements the 'joint' penalty type from
#             the matlab code, with the velocity/velocity diff/etc.
#             penalties removed. (use CostState instead)
#         Args:
#             sample: A single sample.
#         """
#         T = sample.T
#         dX = sample.dX
#         dU = sample.dU
#
#         wpm = get_ramp_multiplier(
#             self.config['ramp_option'], T,
#             wp_final_multiplier=self.config['wp_final_multiplier']
#         )
#         wp = self.config['wp'] * np.expand_dims(wpm, axis=-1)
#
#         # Initialize terms.
#         l = np.zeros(T)
#         lu = np.zeros((T, dU))
#         lx = np.zeros((T, dX))
#         luu = np.zeros((T, dU, dU))
#         lxx = np.zeros((T, dX, dX))
#         lux = np.zeros((T, dU, dX))
#
#         # Choose target.
#         tgt = self.config['target_end_effector']
#         pt = sample.get_X()
#         dist = pt - tgt
#         # TODO - These should be partially zeros so we're not double
#         #        counting.
#         #        (see pts_jacobian_only in matlab costinfos code)
#         jx = sample.get(END_EFFECTOR_POINT_JACOBIANS)
#
#         # Evaluate penalty term. Use estimated Jacobians and no higher
#         # order terms.
#         jxx_zeros = np.zeros((T, dist.shape[1], jx.shape[2], jx.shape[2]))
#         l, ls, lss = self._hyperparams['evalnorm'](
#             wp, dist, jx, jxx_zeros, self._hyperparams['l1'],
#             self._hyperparams['l2'], self._hyperparams['alpha']
#         )
#         # Add to current terms.
#         sample.agent.pack_data_x(lx, ls, data_types=[JOINT_ANGLES])
#         sample.agent.pack_data_x(lxx, lss,
#                                  data_types=[JOINT_ANGLES, JOINT_ANGLES])
#
#         return l, lx, lu, lxx, luu, lux



def get_ramp_multiplier(ramp_option, T, wp_final_multiplier=1.0):
    """
    Return a time-varying multiplier.
    Returns:
        A (T,) float vector containing weights for each time step.
    """
    if ramp_option == RAMP_CONSTANT:
        wpm = np.ones(T)
    elif ramp_option == RAMP_LINEAR:
        wpm = (np.arange(T, dtype=np.float32) + 1) / T
    elif ramp_option == RAMP_QUADRATIC:
        wpm = ((np.arange(T, dtype=np.float32) + 1) / T) ** 2
    elif ramp_option == RAMP_FINAL_ONLY:
        wpm = np.zeros(T)
        wpm[T-1] = 1.0
    else:
        raise ValueError('Unknown cost ramp requested!')
    wpm[-1] *= wp_final_multiplier
    return wpm


def evall1l2term(wp, d, Jd, Jdd, l1, l2, alpha):
    """
    Evaluate and compute derivatives for combined l1/l2 norm penalty.
    loss = (0.5 * l2 * d^2) + (l1 * sqrt(alpha + d^2))
    Args:
        wp: T x D matrix with weights for each dimension and time step.
        d: T x D states to evaluate norm on.
        Jd: T x D x Dx Jacobian - derivative of d with respect to state.
        Jdd: T x D x Dx x Dx Jacobian - 2nd derivative of d with respect
            to state.
        l1: l1 loss weight.
        l2: l2 loss weight.
        alpha: Constant added in square root.
    """
    # Get trajectory length.
    T, _ = d.shape

    # Compute scaled quantities.
    sqrtwp = np.sqrt(wp)
    dsclsq = d * sqrtwp
    dscl = d * wp
    dscls = d * (wp ** 2)

    # Compute total cost.
    l = 0.5 * np.sum(dsclsq ** 2, axis=1) * l2 + \
            np.sqrt(alpha + np.sum(dscl ** 2, axis=1)) * l1

    # First order derivative terms.
    d1 = dscl * l2 + (
        dscls / np.sqrt(alpha + np.sum(dscl ** 2, axis=1, keepdims=True)) * l1
    )
    lx = np.sum(Jd * np.expand_dims(d1, axis=2), axis=1)

    # Second order terms.
    psq = np.expand_dims(
        np.sqrt(alpha + np.sum(dscl ** 2, axis=1, keepdims=True)), axis=1
    )
    d2 = l1 * (
        (np.expand_dims(np.eye(wp.shape[1]), axis=0) *
         (np.expand_dims(wp ** 2, axis=1) / psq)) -
        ((np.expand_dims(dscls, axis=1) *
          np.expand_dims(dscls, axis=2)) / psq ** 3)
    )
    d2 += l2 * (
        np.expand_dims(wp, axis=2) * np.tile(np.eye(wp.shape[1]), [T, 1, 1])
    )

    d1_expand = np.expand_dims(np.expand_dims(d1, axis=-1), axis=-1)
    sec = np.sum(d1_expand * Jdd, axis=1)

    Jd_expand_1 = np.expand_dims(np.expand_dims(Jd, axis=2), axis=4)
    Jd_expand_2 = np.expand_dims(np.expand_dims(Jd, axis=1), axis=3)
    d2_expand = np.expand_dims(np.expand_dims(d2, axis=-1), axis=-1)
    lxx = np.sum(np.sum(Jd_expand_1 * Jd_expand_2 * d2_expand, axis=1), axis=1)

    lxx += 0.5 * sec + 0.5 * np.transpose(sec, [0, 2, 1])

    return l, lx, lxx


def evallogl2term(wp, d, Jd, Jdd, l1, l2, alpha):
    """
    Evaluate and compute derivatives for combined l1/l2 norm penalty.
    loss = (0.5 * l2 * d^2) + (0.5 * l1 * log(alpha + d^2))
    Args:
        wp: T x D matrix with weights for each dimension and time step.
        d: T x D states to evaluate norm on.
        Jd: T x D x Dx Jacobian - derivative of d with respect to state.
        Jdd: T x D x Dx x Dx Jacobian - 2nd derivative of d with respect
            to state.
        l1: l1 loss weight.
        l2: l2 loss weight.
        alpha: Constant added in square root.
    """
    # Get trajectory length.
    T, _ = d.shape

    # Compute scaled quantities.
    sqrtwp = np.sqrt(wp)
    dsclsq = d * sqrtwp
    dscl = d * wp
    dscls = d * (wp ** 2)

    # Compute total cost.
    l = 0.5 * np.sum(dsclsq ** 2, axis=1) * l2 + \
            0.5 * np.log(alpha + np.sum(dscl ** 2, axis=1)) * l1
    # First order derivative terms.
    d1 = dscl * l2 + (
        dscls / (alpha + np.sum(dscl ** 2, axis=1, keepdims=True)) * l1
    )
    lx = np.sum(Jd * np.expand_dims(d1, axis=2), axis=1)

    # Second order terms.
    psq = np.expand_dims(
        alpha + np.sum(dscl ** 2, axis=1, keepdims=True), axis=1
    )
    #TODO: Need * 2.0 somewhere in following line, or * 0.0 which is
    #      wrong but better.
    d2 = l1 * (
        (np.expand_dims(np.eye(wp.shape[1]), axis=0) *
         (np.expand_dims(wp ** 2, axis=1) / psq)) -
        ((np.expand_dims(dscls, axis=1) *
          np.expand_dims(dscls, axis=2)) / psq ** 2)
    )
    d2 += l2 * (
        np.expand_dims(wp, axis=2) * np.tile(np.eye(wp.shape[1]), [T, 1, 1])
    )

    d1_expand = np.expand_dims(np.expand_dims(d1, axis=-1), axis=-1)
    sec = np.sum(d1_expand * Jdd, axis=1)

    Jd_expand_1 = np.expand_dims(np.expand_dims(Jd, axis=2), axis=4)
    Jd_expand_2 = np.expand_dims(np.expand_dims(Jd, axis=1), axis=3)
    d2_expand = np.expand_dims(np.expand_dims(d2, axis=-1), axis=-1)
    lxx = np.sum(np.sum(Jd_expand_1 * Jd_expand_2 * d2_expand, axis=1), axis=1)

    lxx += 0.5 * sec + 0.5 * np.transpose(sec, [0, 2, 1])

    return l, lx, lxx
