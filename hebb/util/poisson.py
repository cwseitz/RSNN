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

    def __init__(self, t, n_in, batches=1, random_select=None, rates=None):

        self.n_in = n_in
        self.nsteps = len(t)
        self.random_select = random_select
        self.t = t
        self.dt = np.mean(np.diff(self.t))
        self.batches = batches
        self.r0 = 20 #default rate (Hz)

        if rates is None:
            rates = self.r0*np.ones((self.n_in, self.batches, self.nsteps))

        self.rates = rates

    def run_generator(self):

        self.r = self.rates*self.dt
        self.x = np.random.uniform(0,1,size=(self.n_in,self.batches,self.nsteps))
        spikes = np.array(self.x < self.r, dtype=np.int32)

        if self.random_select != None:
            rng = default_rng()
            x = rng.choice(self.n_in, size=self.random_select, replace=False)
            spikes[x,:,:] = 0

        return spikes
