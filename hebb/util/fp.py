import numpy as np
from ..models.network import *

class FokkerPlanck1D:

    def __init__(self, v_init, in_weights, rec_weights, input_rates, dtype=np.float32):

        """

        Solve the Fokker-Planck equation in the 1-dimensional case (a single LIF neuron)
        numerically for the voltage probability density as a function of time.

        dP/dt =

        Parameters
        ----------
        v_init : ndarray
            Initial probability distribution over the voltages
        in_weights: ndarray
            Input connectivity
        rec_weights: ndarray
            Recurrent connectivity
        input_rates: ndarray
            Instantaneous firing rate. Has shape (input_units, nsteps)

        """

        #Basic parameters
        self.v_init = v_init

    def solve(self, rates):

        for t in range(1, self.nsteps):
            self.pdf[:,:,t] = self.pdf[:,:,t-1]
