import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import cm
from .network import *
from ..util import *

################################################################################
##
##    All models inherit from base model Neuron the
##    common params, functions. Individual neuron models
##    define their specific state variables. Simulations of N = [1,inf)
##    neurons can be run in two different ways:
##
##    (1) - A bipartite architecture where input spikes
##       and a synaptic conductance matrix are generated
##       by the user and passed as arguments to the Neuron
##       constructor. Input currents are a weighted sum of spikes
##       where the weights come from the synaptic conductance matrix
##
##    (2) - N = [1,inf) neurons are simulated that are not connected
##        but the input current to each neuron is provided as a
##        function of time. Useful to probe the response of different
##        models to stochastic or deterministic input currents and compare
##        the predictions of statistical models e.g. Fokker-Planck equations
##
##     **Note: Does not currently support synaptic conductance matrices
##             that are a function of time. Do not try to use the same object
##             for both (1) and (2). Re-initialize.
##
##     State variables are 3D tensors with shape (N, trials, time)
##
################################################################################

################################################################################
## Author: Clayton Seitz
## Copyright: 2021, The Hebb Project
## Email: cwseitz@uchicago.edu
################################################################################

class Neuron:

    def __init__(self, T, dt, tau_ref, J, trials=1, dtype=np.float32):

        """

        Neuron base class

        Parameters
        ----------

        T: float
            Total simulation time in seconds
        dt: float
            Time resolution in seconds
        tau_ref : float
            Refractory time in seconds
        J : 2D ndarray
            Synaptic connectivity matrix
        trials : int
            Number of stimulations to run
        dtype : numpy data type
            Data type to use for neuron state variables

        """

        #Basic parameters common to all neurons
        self.dt = dt #time resolution
        self.T = T #simulation period
        self.trials = trials #number of trials
        self.tau_ref = tau_ref #refractory period
        self.nsteps = 1 + int(round((self.T/dt))) #number of 'cuts'
        self.ref_steps = int(self.tau_ref/self.dt) #number of steps for refractory period
        self.J = J #synaptic connectivity
        self.N = self.J.shape[0]
        self.dtype = dtype #data type
        self.shape = (self.N,self.trials,self.nsteps)

class ClampedLIF(Neuron):

    def __init__(self,  T, dt, tau_ref, J, trials=1, tau=0.02, g_l=8.75, thr=20, dtype=np.float32):

        super(ClampedLIF, self).__init__(T, dt, tau_ref, J=J, trials=trials, dtype=dtype)

        """

        Leaky Integrate & Fire (LIF) neuron model where a subset of the
        neurons are clamped to user specified spike trains. This is useful
        when you want an 'input population' to be part of the larger network

        Parameters
        ----------

        T: float
            Total simulation time in seconds
        dt: float
            Time resolution in seconds
        tau_ref : float
            Refractory time in seconds
        J : 2D ndarray
            Synaptic connectivity matrix
        trials : int
            Number of stimulations to run
        tau : float
            Membrane time constant
        g_l : float
            The leak conductance of the membrane
        thr : float
            Firing threshold in mV
        dtype : numpy data type
            Data type to use for neuron state variables

        """

        #ClampedLIF specific parameters
        self.tau = tau
        self.g_l = g_l
        self.thr = thr

    def spike_function(self, v):
        z = (v >= self.thr).astype('int')
        return z

    def zero_state(self):

        #Initialize state variables
        self.I = np.zeros(shape=(self.M, self.trials, self.nsteps), dtype=self.dtype)
        self.V = np.zeros(shape=(self.M, self.trials, self.nsteps), dtype=self.dtype)
        self.R = np.zeros(shape=(self.M,self.trials,self.nsteps+self.ref_steps), dtype=np.bool)
        self.Z = np.zeros(shape=(self.N, self.trials, self.nsteps), dtype=np.bool)

    def call(self, spikes, clamp_idx):

        """

        This function will infer the indices of clamped neurons based
        on the first axis of 'spikes'. The user is responsible for
        determining which neurons are clamped (e.g., random, a group, etc.)

        spikes : 3D ndarray
            Used to clamp the observable state Z of specified neurons, often
            to use a subnetwork as an 'input population'.
        clamp_idx : 3D ndarray
            Indices along first axis indicating which neurons are clamped
        """

        self.spikes = spikes
        self.clamp = np.zeros((self.N,))
        self.clamp[clamp_idx] = 1
        self.clamp = np.mod(self.clamp + 1,2) #invert clamp (see usage below)
        self.no_clamp_idx = np.argwhere(self.clamp > 0)
        self.no_clamp_idx = self.no_clamp_idx.reshape((self.no_clamp_idx.shape[0],))
        self.M = len(self.no_clamp_idx)
        self.J = self.J[self.no_clamp_idx,:]
        self.zero_state()

        start, end = 1, self.nsteps

        for i in range(start, end):

            #enforce the clamp
            self.Z[:,:,i-1] = np.einsum('ij,i -> ij', self.Z[:,:,i-1], self.clamp) + self.spikes[:,:,i-1]
            self.I[:,:,i] =  np.matmul(self.J, self.Z[:,:,i-1])
            #apply spike function to previous time step
            self.Z[self.no_clamp_idx,:,i] = self.spike_function(self.V[:,:,i-1])
            #check if the neuron spiked in the last tau_ref time steps
            self.R[:,:,i+self.ref_steps] = np.sum(self.Z[self.no_clamp_idx,:,i-self.ref_steps:i+1], axis=-1)
            self.V[:,:,i] = self.V[:,:,i-1] - self.dt*self.V[:,:,i-1]/self.tau +\
                            self.I[:,:,i-1]/(self.tau*self.g_l)
            #Enforce refractory period
            self.V[:,:,i] = self.V[:,:,i] - self.V[:,:,i]*self.R[:,:,i+self.ref_steps]

class LIF(Neuron):

    def __init__(self,  T, dt, tau_ref, J, trials=1, tau=0.02, g_l=20, thr=20, dtype=np.float32):

        super(LIF, self).__init__(T, dt, tau_ref, J=J, trials=trials, dtype=dtype)

        """
        Basic Leaky Integrate & Fire (LIF) neuron model. For use when the
        input currents to each neuron (from an external input pop) are known.
        To generate currents from spikes and an input connectivity matrix,
        see utility functions.

        Parameters
        ----------

        T: float
            Total simulation time in seconds
        dt: float
            Time resolution in seconds
        tau_ref : float
            Refractory time in seconds
        J : 2D ndarray
            Synaptic connectivity matrix
        trials : int
            Number of stimulations to run
        tau : float
            Membrane time constant
        g_l : float
            The leak conductance of the membrane
        thr : float
            Firing threshold
        dtype : numpy data type
            Data type to use for neuron state variables

        """

        #LIF specific parameters
        self.tau = tau
        self.g_l = g_l
        self.thr = thr

    def spike_function(self, v):
        z = (v >= self.thr).astype('int')
        return z

    def zero_state(self):

        #Initialize state variables
        self.I = np.zeros(shape=(self.N, self.trials, self.nsteps), dtype=self.dtype)
        self.V = np.zeros(shape=(self.N, self.trials, self.nsteps), dtype=self.dtype)
        self.Z = np.zeros(shape=(self.N, self.trials, self.nsteps), dtype=np.int8)
        self.R = np.zeros(shape=(self.N,self.trials,self.nsteps+self.ref_steps), dtype=np.int8)

    def check_shape(self, x):
        if x is None:
            raise ValueError('Input object was not set')
        else:
            if x.shape != self.shape:
                raise ValueError('Check input object shape')
        return True

    def call(self, currents):

        self.currents = currents
        self.check_shape(self.currents)
        self.zero_state()

        start, end = self.ref_steps, self.nsteps
        for i in range(start, end):
            i_in = self.currents[:,:,i-1]
            i_re = np.matmul(self.J, self.Z[:,:,i-1])
            self.I[:,:,i] =  i_in + i_re
            #apply spike function to previous time step
            self.Z[:,:,i] = self.spike_function(self.V[:,:,i-1])
            #check if the neuron spiked in the last tau_ref time steps
            self.R[:,:,i+self.ref_steps] = np.sum(self.Z[:,:,i-self.ref_steps:i+1], axis=-1)
            self.V[:,:,i] = self.V[:,:,i-1] - self.dt*self.V[:,:,i-1]/self.tau +\
                            self.I[:,:,i-1]/(self.tau*self.g_l)
            #Enforce refractory period
            self.V[:,:,i] = self.V[:,:,i] - self.V[:,:,i]*self.R[:,:,i+self.ref_steps]
