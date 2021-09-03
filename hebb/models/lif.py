import numpy as np
import tensorflow as tf
from .conn import *

class ExInLIF:

    def __init__(self, n_in, n_rec, p_xx, period=100, p_e=0.8, tau=20.,
                 thr=0.615, dt=1., tau_ref=3, batches=1, dtype=np.float32):

        """

        RNN cell model to simulate a network of
        Leaky Integrate and Fire (LIF) neurons.

        Parameters
        ----------
        n_in : int
            Number of input neurons
        n_rec : int
            Number of recurrent neurons
        p_xx : ndarray
            Matrix of [[e->i, i->e], [e->e, and i->i]] connection probabilties
        p_e : float
            Probability of a neuron being excitatory
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
        self.n_rec = n_rec
        self.n_excite = int(round(p_e*n_rec))
        self.n_inhib = self.n_rec - self.n_excite
        self.p_xx = p_xx
        self.dtype = dtype
        self.tau = tf.constant(tau, dtype=dtype)
        self.tau_ref = tau_ref
        self.decay = tf.exp(-dt / self.tau)
        self.thr = thr
        self.period = period
        self.batches = batches

        #Network connectivity
        self.in_cmg = InputConnectivityGenerator(n_in, n_rec)
        ex_in_params = [self.n_excite, self.n_inhib, self.p_xx]
        self.rec_cmg = ExInConnectivityMatrixGenerator(*ex_in_params)
        self.in_weights = self.in_cmg.run_generator()
        self.rec_weights = self.rec_cmg.run_generator()
        self.zero_state(self.batches)

    def spike_function(self, v):

        """
        Thresholds the voltage vector to update the observable state tensor
        """

        z_ = np.greater_equal(v, self.thr)
        z = z_.astype('int32')

        return z


    def zero_state(self, batches=1):

        #pad along time axis for calculating the refractory variable as a sum over z
        self.v = np.zeros(shape=(self.n_rec, batches, self.period+self.tau_ref), dtype=self.dtype)
        self.z = np.zeros(shape=(self.n_rec, batches, self.period+self.tau_ref), dtype=np.int8)
        self.r = np.zeros(shape=(self.n_rec, batches, self.period+self.tau_ref), dtype=np.int8)

    def call(self, input):

        for t in range(self.tau_ref, self.period):

            #check if the neuron spiked in the last tau_ref time steps
            self.r[:,:,t] = np.sum(self.z[:,:,t-self.tau_ref:t], axis=-1)

            #integrate input and recurrent currents from spikes at previous time step
            i_in = np.matmul(self.in_weights, input[:,:,t-1])
            i_rec = np.matmul(self.rec_weights, self.z[:,:,t-1])

            #enforce the refractory period
            i_reset = -(self.v[:,:,t-1] + i_in + i_rec)*self.r[:,:,t]

            #update the voltage
            self.v[:,:,t] = self.decay*self.v[:,:,t-1] + i_in + i_rec + i_reset

            #apply spike function to current time step
            self.z[:,:,t] = self.spike_function(self.v[:,:,t])

        #truncate zero padding for tau_ref
        state = (self.v[:,:,self.tau_ref:], self.z[:,:,self.tau_ref:], self.r[:,:,self.tau_ref:])
        return state
