import numpy as np
from ..models.conn import *

class FokkerPlanck1D:

    def __init__(self, n_in, nsteps=100, tau=20., thr=0.615,
                 dt=0.01, tau_ref=3, batches=1, dtype=np.float32):

        """

        Solve the Fokker-Planck equations in the 1-dimensional case (a single LIF neuron)
        numerically for the voltage probability density as a function of time.

        Parameters
        ----------
        n_in : int
            Number of input neurons
        tau : float
            Membrane time constant (a.u.)
        thr : float
            Spiking voltage threshold
        dt : float
            Time resolution
        dtype: numpy datatype
            Standard datatype for objects during the simulation

        """

        #Basic parameters
        self.dt = dt
        self.n_in = n_in
        self.dtype = dtype
        self.tau = tau
        self.tau_ref = tau_ref
        self.decay = np.exp(-dt/self.tau)
        self.thr = thr
        self.nsteps = nsteps
        self.batches = batches
        self.n_rec = 1
        self.vmax = 2*self.thr
        self.dv = np.exp(dt/self.tau)

        #Network connectivity
        self.in_cmg = InputConnectivityGenerator(n_in, self.n_rec)
        self.zero_state()

    def spike_function(self, v):

        """
        Thresholds the voltage vector to update the observable state tensor
        """

        z_ = np.greater_equal(v, self.thr)
        z = z_.astype('int32')

        return z


    def zero_state(self):

        """
        Initializes a delta function at resting potential
        ***temporarily ignoring refractoriness
        """

        #pad along time axis for calculating the refractory variable as a sum over z
        nv = int(round(self.vmax/self.dv))
        self.pdf = np.zeros(shape=(self.n_rec, nv, self.nsteps), dtype=self.dtype)
        self.bins = np.linspace(0, self.vmax, nv)
        self.pdf[:,0,0] = 1 #delta function at resting potential

    def solve(self, rates):

        for t in range(1, self.nsteps):
            self.pdf[:,:,t] = self.pdf[:,:,t-1]
