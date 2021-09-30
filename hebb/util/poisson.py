import numpy as np
from numpy.random import default_rng

class Poisson:

    """

    Generate an ensemble of Poisson spike trains from a vector of firing rates

    Parameters
    ----------
    rates : ndarray
        A matrix - each value providing the firing rate for a single input unit
        By default, generate an ensemble of homogeneous poisson processes
        with rate 20Hz

    Returns
    -------
    spikes : 2d ndarray
        a binary matrix containing the spiking patterns for the ensemble
        (i, j) --> (unit, nsteps)

    """

    def __init__(self, T, dt, N, trials=1, rates=None, random_select=None):

        self.T = T
        self.dt = dt
        self.N = N
        self.nsteps = 1 + int(round(T/dt))
        self.random_select = random_select
        self.trials = trials

        if rates is None:
            self.r0 = 20 #default rate (Hz)
            rates = self.r0*np.ones((self.N, self.trials, self.nsteps))

        self.rates = rates

    def run_generator(self):

        self.r = self.rates*self.dt
        self.x = np.random.uniform(0,1,size=(self.N,self.trials,self.nsteps))
        spikes = np.array(self.x < self.r, dtype=np.int32)

        if self.random_select != None:
            rng = default_rng()
            x = rng.choice(self.N, size=self.random_select, replace=False)
            spikes[x,:,:] = 0

        return spikes
