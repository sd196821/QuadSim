import pickle
import os
import uuid
import copy
import tensorflow as tf
import numpy as np
import scipy as sp

class policy_NN():
    """
    Policy in Neural Networks of Tensorflow
    """

    def __init__(self, dU, obs_tensor, act_op, feat_op, var, sess, device_string, copy_param_scope=None):
        self.dU = dU
        self.obs_tensor = obs_tensor
        self.act_op = act_op
        self.feat_op = feat_op
        self.sess = sess
        self.device_string = device_string
        self.chol_pol_covar = np.diag(np.sqrt(var))
        self.scale = None  # must be set from elsewhere based on observations
        self.bias = None
        self.x_idx = None

        if copy_param_scope:
            self.copy_params = tf.get_collection(tf.GraphKeys.VARIABLES, scope=copy_param_scope)
            self.copy_params_assign_placeholders = [tf.placeholder(tf.float32, shape=param.get_shape()) for
                                                    param in self.copy_params]

            self.copy_params_assign_ops = [tf.assign(self.copy_params[i],
                                                     self.copy_params_assign_placeholders[i])
                                           for i in range(len(self.copy_params))]

    def act(self, x, obs, t, noise):
        """
        Return an action for a state.
        Args:
            x: State vector.
            obs: Observation vector.
            t: Time step.
            noise: Action noise. This will be scaled by the variance.
        """

        # Normalize obs.
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, axis=0)
        obs[:, self.x_idx] = obs[:, self.x_idx].dot(self.scale) + self.bias
        with tf.device(self.device_string):
            action_mean = self.sess.run(self.act_op, feed_dict={self.obs_tensor: obs})
        if noise is None:
            u = action_mean
        else:
            u = action_mean + self.chol_pol_covar.T.dot(noise)
        return u[0]  # the DAG computations are batched by default, but we use batch size 1.

    def get_features(self, obs):
        """
        Return the image features for an observation.
        Args:
            obs: Observation vector.
        """
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, axis=0)
        # Assume that features don't depend on the robot config, so don't normalize by scale and bias.
        with tf.device(self.device_string):
            feat = self.sess.run(self.feat_op, feed_dict={self.obs_tensor: obs})
        return feat[0]  # the DAG computations are batched by default, but we use batch size 1.

    def get_copy_params(self):
        param_values = self.sess.run(self.copy_params)
        return {self.copy_params[i].name: param_values[i] for i in range(len(self.copy_params))}

    def set_copy_params(self, param_values):
        value_list = [param_values[self.copy_params[i].name] for i in range(len(self.copy_params))]
        feeds = {self.copy_params_assign_placeholders[i]: value_list[i] for i in range(len(self.copy_params))}
        self.sess.run(self.copy_params_assign_ops, feed_dict=feeds)

    def pickle_policy(self, deg_obs, deg_action, checkpoint_path, goal_state=None, should_hash=False):
        """
        We can save just the policy if we are only interested in running forward at a later point
        without needing a policy optimization class. Useful for debugging and deploying.
        """
        if should_hash is True:
            hash_str = str(uuid.uuid4())
            checkpoint_path += hash_str
        os.mkdir(checkpoint_path + '/')
        checkpoint_path += '/_pol'
        pickled_pol = {'deg_obs': deg_obs, 'deg_action': deg_action, 'chol_pol_covar': self.chol_pol_covar,
                       'checkpoint_path_tf': checkpoint_path + '_tf_data', 'scale': self.scale, 'bias': self.bias,
                       'device_string': self.device_string, 'goal_state': goal_state, 'x_idx': self.x_idx}
        pickle.dump(pickled_pol, open(checkpoint_path, "wb"))
        saver = tf.train.Saver()
        saver.save(self.sess, checkpoint_path + '_tf_data')

    @classmethod
    def load_policy(cls, policy_dict_path, tf_generator, network_config=None):
        """
        For when we only need to load a policy for the forward pass. For instance, to run on the robot from
        a checkpointed policy.
        """
        from tensorflow.python.framework import ops
        ops.reset_default_graph()  # we need to destroy the default graph before re_init or checkpoint won't restore.
        pol_dict = pickle.load(open(policy_dict_path, "rb"))
        tf_map = tf_generator(dim_input=pol_dict['deg_obs'], dim_output=pol_dict['deg_action'],
                              batch_size=1, network_config=network_config)

        sess = tf.Session()
        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        saver = tf.train.Saver()
        check_file = pol_dict['checkpoint_path_tf']
        saver.restore(sess, check_file)

        device_string = pol_dict['device_string']

        cls_init = cls(pol_dict['deg_action'], tf_map.get_input_tensor(), tf_map.get_output_op(), np.zeros((1,)),
                       sess, device_string)
        cls_init.chol_pol_covar = pol_dict['chol_pol_covar']
        cls_init.scale = pol_dict['scale']
        cls_init.bias = pol_dict['bias']
        cls_init.x_idx = pol_dict['x_idx']
        return cls_init


class policy_LQG():
    """
    Time-varying linear Gaussian policy.
    U = K*x + k + noise, where noise ~ N(0, chol_pol_covar)
    """

    def __init__(self, K, k, pol_covar, chol_pol_covar, inv_pol_covar):
        # Assume K has the correct shape, and make sure others match.
        self.T = K.shape[0]
        self.dU = K.shape[1]
        self.dX = K.shape[2]

        self.K = K
        self.k = k
        self.pol_covar = pol_covar
        self.chol_pol_covar = chol_pol_covar
        self.inv_pol_covar = inv_pol_covar

    def act(self, x, obs, t, noise=None):
        """
        Return an action for a state.
        Args:
            x: State vector.
            obs: Observation vector.
            t: Time step.
            noise: Action noise. This will be scaled by the variance.
        """
        u = self.K[t].dot(x) + self.k[t]
        u += self.chol_pol_covar[t].T.dot(noise)
        return u

    def fold_k(self, noise):
        """
        Fold noise into k.
        Args:
            noise: A T x Du noise vector with mean 0 and variance 1.
        Returns:
            k: A T x dU bias vector.
        """
        k = np.zeros_like(self.k)
        for i in range(self.T):
            scaled_noise = self.chol_pol_covar[i].T.dot(noise[i])
            k[i] = scaled_noise + self.k[i]
        return k

    def nans_like(self):
        """
        Returns:
            A new linear Gaussian policy object with the same dimensions
            but all values filled with NaNs.
        """
        policy = policy_LQG(
            np.zeros_like(self.K), np.zeros_like(self.k),
            np.zeros_like(self.pol_covar), np.zeros_like(self.chol_pol_covar),
            np.zeros_like(self.inv_pol_covar)
        )
        policy.K.fill(np.nan)
        policy.k.fill(np.nan)
        policy.pol_covar.fill(np.nan)
        policy.chol_pol_covar.fill(np.nan)
        policy.inv_pol_covar.fill(np.nan)
        return policy

def init_lqr(INIT_LQG):
    """
    Return initial gains for a time-varying linear Gaussian controller
    that tries to hold the initial position.
    """
    config = copy.deepcopy(INIT_LQG)

    x0, dX, dU = config['x0'], config['dX'], config['dU']
    dt, T = config['dt'], config['T']

    # Notation notes:
    # L = loss, Q = q-function (dX+dU dimensional),
    # V = value function (dX dimensional), F = dynamics
    # Vectors are lower-case, matrices are upper case.
    # Derivatives: x = state, u = action, t = state+action (trajectory).
    # The time index is denoted by _t after the above.
    # Ex. Ltt_t = Loss, 2nd derivative (w.r.t. trajectory),
    # indexed by time t.

    # Constants.
    idx_x = slice(dX)  # Slices out state.
    idx_u = slice(dX, dX+dU)  # Slices out actions.

    if len(config['init_acc']) == 0:
        config['init_acc'] = np.zeros(dU)

    if len(config['init_gains']) == 0:
        config['init_gains'] = np.ones(dU)

    # Set up simple linear dynamics model.
    Fd, fc = guess_dynamics(config['init_gains'], config['init_acc'],
                            dX, dU, dt)

    # Setup a cost function based on stiffness.
    # Ltt = (dX+dU) by (dX+dU) - Hessian of loss with respect to
    # trajectory at a single timestep.
    Ltt = np.diag(np.hstack([
        config['stiffness'] * np.ones(dU),
        config['stiffness'] * config['stiffness_vel'] * np.ones(dU),
        np.zeros(dX - dU*2), np.ones(dU)
    ]))
    Ltt = Ltt / config['init_var']  # Cost function - quadratic term.
    lt = -Ltt.dot(np.r_[x0, np.zeros(dU)])  # Cost function - linear term.

    # Perform dynamic programming.
    K = np.zeros((T, dU, dX))  # Controller gains matrix.
    k = np.zeros((T, dU))  # Controller bias term.
    PSig = np.zeros((T, dU, dU))  # Covariance of noise.
    cholPSig = np.zeros((T, dU, dU))  # Cholesky decomposition.
    invPSig = np.zeros((T, dU, dU))  # Inverse of covariance.
    vx_t = np.zeros(dX)  # Vx = dV/dX. Derivative of value function.
    Vxx_t = np.zeros((dX, dX))  # Vxx = ddV/dXdX.

    for t in range(T - 1, -1, -1):
        # Compute Q function at this step.
        if t == (T - 1):
            Ltt_t = config['final_weight'] * Ltt
            lt_t = config['final_weight'] * lt
        else:
            Ltt_t = Ltt
            lt_t = lt
        # Qtt = (dX+dU) by (dX+dU) 2nd Derivative of Q-function with
        # respect to trajectory (dX+dU).
        Qtt_t = Ltt_t + Fd.T.dot(Vxx_t).dot(Fd)
        # Qt = (dX+dU) 1st Derivative of Q-function with respect to
        # trajectory (dX+dU).
        qt_t = lt_t + Fd.T.dot(vx_t + Vxx_t.dot(fc))

        # Compute preceding value function.
        U = sp.linalg.cholesky(Qtt_t[idx_u, idx_u])
        L = U.T

        invPSig[t, :, :] = Qtt_t[idx_u, idx_u]
        PSig[t, :, :] = sp.linalg.solve_triangular(
            U, sp.linalg.solve_triangular(L, np.eye(dU), lower=True)
        )
        cholPSig[t, :, :] = sp.linalg.cholesky(PSig[t, :, :])
        K[t, :, :] = -sp.linalg.solve_triangular(
            U, sp.linalg.solve_triangular(L, Qtt_t[idx_u, idx_x], lower=True)
        )
        k[t, :] = -sp.linalg.solve_triangular(
            U, sp.linalg.solve_triangular(L, qt_t[idx_u], lower=True)
        )
        Vxx_t = Qtt_t[idx_x, idx_x] + Qtt_t[idx_x, idx_u].dot(K[t, :, :])
        vx_t = qt_t[idx_x] + Qtt_t[idx_x, idx_u].dot(k[t, :])
        Vxx_t = 0.5 * (Vxx_t + Vxx_t.T)

    return policy_LQG(K, k, PSig, cholPSig, invPSig)

def init_pd(INIT_PD):
    """
    This function initializes the linear-Gaussian controller as a
    proportional-derivative (PD) controller with Gaussian noise. The
    position gains are controlled by the variable pos_gains, velocity
    gains are controlled by pos_gains*vel_gans_mult.
    """
    config = copy.deepcopy(INIT_PD)

    dU, dQ, dX = config['dU'], config['dQ'], config['dX']
    x0, T = config['x0'], config['T']

    # Choose initialization mode.
    Kp = 1.0
    Kv = config['vel_gains_mult']
    if dU < dQ:
        K = -config['pos_gains'] * np.tile(
            [np.eye(dU) * Kp, np.zeros((dU, dQ-dU)),
             np.eye(dU) * Kv, np.zeros((dU, dQ-dU))],
            [T, 1, 1]
        )
    else:
        K = -config['pos_gains'] * np.tile(
            np.hstack([
                np.eye(dU) * Kp, np.eye(dU) * Kv,
                np.zeros((dU, dX - dU*2))
            ]), [T, 1, 1]
        )
    k = np.tile(-K[0, :, :].dot(x0), [T, 1])
    PSig = config['init_var'] * np.tile(np.eye(dU), [T, 1, 1])
    cholPSig = np.sqrt(config['init_var']) * np.tile(np.eye(dU), [T, 1, 1])
    invPSig = (1.0 / config['init_var']) * np.tile(np.eye(dU), [T, 1, 1])

    return policy_LQG(K, k, PSig, cholPSig, invPSig)


def guess_dynamics(gains, acc, dX, dU, dt):
    """
    Initial guess at the model using position-velocity assumption.
    Note: This code assumes joint positions occupy the first dU state
          indices and joint velocities occupy the next dU.
    Args:
        gains: dU dimensional joint gains.
        acc: dU dimensional joint acceleration.
        dX: Dimensionality of the state.
        dU: Dimensionality of the action.
        dt: Length of a time step.
    Returns:
        Fd: A dX by dX+dU transition matrix.
        fc: A dX bias vector.
    """
    Fd = np.vstack([
        np.hstack([
            np.eye(dU), dt * np.eye(dU), np.zeros((dU, dX - dU*2)),
            dt ** 2 * np.diag(gains)
        ]),
        np.hstack([
            np.zeros((dU, dU)), np.eye(dU), np.zeros((dU, dX - dU*2)),
            dt * np.diag(gains)
        ]),
        np.zeros((dX - dU*2, dX+dU))
    ])
    fc = np.hstack([acc * dt ** 2, acc * dt, np.zeros((dX - dU*2))])
    return Fd, fc


