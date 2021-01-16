import numpy as np

class sample():
    def __init__(self, T, dX, dU, dO=None, dM=None):
        self.T = T
        self.dX = dX
        self.dU = dU
        self.dO = dO
        self.dM = dM

        self.data = []
        self.X = np.empty((self.T, self.dX))
        self.X.fill(np.nan)
        self.obs = np.empty((self.T, self.dO))
        self.obs.fill(np.nan)
        self.meta = np.empty(self.dM)
        self.meta.fill(np.nan)

    def set(self, sensor_data, t=None):
        """ Set trajectory data for a particular sensor. """
        if t is None:
            self.data = sensor_data
            self.X.fill(np.nan)  # Invalidate existing X.
            self.obs.fill(np.nan)  # Invalidate existing obs.
            self.meta.fill(np.nan)  # Invalidate existing meta data.
        else:
            self.data[t, :] = sensor_data
            self.X[t, :].fill(np.nan)
            self.obs[t, :].fill(np.nan)

    def get(self, t=None):
        """ Get trajectory data for a particular sensor. """
        return (self.data if t is None
                else self.data[t, :])

    def get_X(self, t=None):
        """ Get the state. Put it together if not precomputed. """
        X = self.X if t is None else self.X[t, :]
        if np.any(np.isnan(X)):
                data = (self.data if t is None
                        else self.data[t, :])
                X = data
        return X

