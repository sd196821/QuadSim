import numpy as np

class sample():
    def __init__(self, T, dX, dU, dO=None, dM):
        self.T =  T
        self.dX = dX
        self.dU = dU
        self.dO = dO
        self.dM = dM

        self.data= []
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
