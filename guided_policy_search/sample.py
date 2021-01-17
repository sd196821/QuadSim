import numpy as np
import scipy.ndimage as sp_ndimage

class sample():
    def __init__(self, T, dX, dU, dO=None, dM=None):
        self.T = T
        self.dX = dX
        self.dU = dU
        self.dO = dO
        self.dM = dM

        self.X = np.empty((self.T, self.dX))
        self.X.fill(np.nan)
        self.U = np.empty((self.T, self.dU))
        self.U.fill(np.nan)
        self.obs = np.empty((self.T, self.dO))
        self.obs.fill(np.nan)
        self.meta = np.empty(self.dM)
        self.meta.fill(np.nan)

    def set(self, X=None, U=None, obs=None, t=None):
        """ Set trajectory data for a particular sensor. """
        if t is None:
            self.X.fill(np.nan)  # Invalidate existing X.
            self.obs.fill(np.nan)  # Invalidate existing obs.
            self.U.fill(np.nan)
            self.meta.fill(np.nan)  # Invalidate existing meta data.
        else:
            self.X[t, :] = X
            self.obs[t, :] = obs
            self.U[t, :] = U

    def get_X(self, t=None):
        """ Get the state. Put it together if not precomputed. """
        X = self.X if t is None else self.X[t, :]
        return X

    def get_U(self, t=None):
        """ Get the state. Put it together if not precomputed. """
        U = self.U if t is None else self.U[t, :]
        return U

    def get_obs(self, t=None):
        """ Get the state. Put it together if not precomputed. """
        obs = self.obs if t is None else self.obs[t, :]
        return obs

    def generate_noise(self, smooth, var, renorm):
        """
        Generate a T x dU gaussian-distributed noise vector. This will
        approximately have mean 0 and variance 1, ignoring smoothing.

        Args:
            T: Number of time steps.
            dU: Dimensionality of actions.
        Hyperparams:
            smooth: Whether or not to perform smoothing of noise.
            var : If smooth=True, applies a Gaussian filter with this
                variance.
            renorm : If smooth=True, renormalizes data to have variance 1
                after smoothing.
        """
        dU = self.dU
        T = self.T
        noise = np.random.randn(T, dU)
        if smooth:
            # Smooth noise. This violates the controller assumption, but
            # might produce smoother motions.
            for i in range(dU):
                noise[:, i] = sp_ndimage.filters.gaussian_filter(noise[:, i], var)
            if renorm:
                variance = np.var(noise, axis=0)
                noise = noise / np.sqrt(variance)
        return noise


class SampleList():
    """ Class that handles writes and reads to sample data. """
    def __init__(self, samples):
        self._samples = samples

    def get_X(self, idx=None):
        """ Returns N x T x dX numpy array of states. """
        if idx is None:
            idx = range(len(self._samples))
        return np.asarray([self._samples[i].get_X() for i in idx])

    def get_U(self, idx=None):
        """ Returns N x T x dU numpy array of actions. """
        if idx is None:
            idx = range(len(self._samples))
        return np.asarray([self._samples[i].get_U() for i in idx])

    def get_noise(self, idx=None):
        """ Returns N x T x dU numpy array of noise generated during rollouts. """
        if idx is None:
            idx = range(len(self._samples))
        return np.asarray([self._samples[i].get(NOISE) for i in idx])

    def get_obs(self, idx=None):
        """ Returns N x T x dO numpy array of features. """
        if idx is None:
            idx = range(len(self._samples))
        return np.asarray([self._samples[i].get_obs() for i in idx])

    def get_samples(self, idx=None):
        """ Returns N sample objects. """
        if idx is None:
            idx = range(len(self._samples))
        return [self._samples[i] for i in idx]

    def num_samples(self):
        """ Returns number of samples. """
        return len(self._samples)

    # Convenience methods.
    def __len__(self):
        return self.num_samples()

    def __getitem__(self, idx):
        return self.get_samples([idx])[0]
