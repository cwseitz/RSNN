import numpy as np

class Poisson:

    """

    Generate an ensemble of Poisson spike trains from a vector of firing rates

    Parameters
    ----------
    rates : ndarray
        A matrix - each value providing the firing rate for a single input unit
        By default, generate an ensemble of homogeneous poisson processes
        with rate equal to 20Hz

    Returns
    -------
    spikes : 2d ndarray
        a binary matrix containing the spiking patterns for the ensemble
        (i, j) --> (unit, nsteps)

    """

    def __init__(self, n_in, nsteps, dt=0.01, batches=1, rates=None):

        self.n_in = n_in
        self.nsteps = nsteps
        self.dt = dt #seconds
        self.batches = batches
        self.def_rate = 20 #default rate (Hz)

        if rates is None:
            rates = self.def_rate*np.ones((self.n_in, self.batches, self.nsteps))

        self.rates = rates

    def run_generator(self):

        self.r = self.rates*self.dt
        self.x = np.random.uniform(0,1,size=(self.n_in,self.batches,self.nsteps))
        spikes = np.array(self.x < self.r, dtype=np.int32)

        return spikes
