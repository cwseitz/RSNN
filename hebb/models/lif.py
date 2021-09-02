import numpy as np
import tensorflow as tf
from .conn import *

class ExInLIF:

    def __init__(self, n_in, n_rec, p_xx, p_e=0.8, tau=20.,
                 thr=0.615, dt=1., dtype=np.float32):

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
        self.data_type = dtype
        self.tau = tf.constant(tau, dtype=dtype)
        self.decay = tf.exp(-dt / self.tau)
        self.thr = thr

        #Network connectivity
        self.in_cmg = InputConnectivityGenerator(n_in, n_rec)
        ex_in_params = [self.n_excite, self.n_inhib, self.p_xx]
        self.rec_cmg = ExInConnectivityMatrixGenerator(*ex_in_params)


    def zero_state(self, batch_size):

        self.v_e = np.zeros(shape=(batch_size, self.n_excite), dtype=self.dtype)
        self.v_i = np.zeros(shape=(batch_size, self.n_inhib), dtype=self.dtype)
        self.z_e = np.zeros(shape=(batch_size, self.n_excite), dtype=self.dtype)
        self.z_i = np.zeros(shape=(batch_size, self.n_inhib), dtype=self.dtype)

    # def call(self, input):
