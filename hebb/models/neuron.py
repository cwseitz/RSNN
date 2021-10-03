import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pims
from .network import *
from ..util import *
from matplotlib import cm
from skimage.io import imsave

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

    def __init__(self,  T, dt, tau_ref, J, trials=1, tau=0.02, g_l=75, thr=20, dtype=np.float32):

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
        self.I = np.zeros(shape=self.shape, dtype=self.dtype)
        self.V = np.zeros(shape=self.shape, dtype=self.dtype)
        self.Z = np.zeros(shape=self.shape, dtype=np.int8)
        self.R = np.zeros(shape=(self.N,self.trials,self.nsteps+self.ref_steps), dtype=np.int8)

    def call(self, spikes, clamp):

        """
        spikes : 3D ndarray
            Used to clamp the observable state Z of specified neurons, often
            to use a subnetwork as an 'input population'.
        clamp : 3D ndarray
            A binary tensor where a value of '1' indicates that neuron of
            a particular batch is clamped at that time
        """

        self.spikes = spikes
        self.clamp = np.mod(clamp + 1,2) #invert clamp (see usage below)
        self.zero_state()

        start, end = 1, self.nsteps

        for i in range(start, end):

            #enforce the clamp
            self.Z[:,:,i-1] = self.Z[:,:,i-1]*self.clamp[:,:,i-1] + self.spikes[:,:,i-1]
            self.I[:,:,i] =  np.matmul(self.J, self.Z[:,:,i-1])
            #apply spike function to previous time step
            self.Z[:,:,i] = self.spike_function(self.V[:,:,i-1])
            #check if the neuron spiked in the last tau_ref time steps
            self.R[:,:,i+self.ref_steps] = np.sum(self.Z[:,:,i-self.ref_steps:i+1], axis=-1)
            self.V[:,:,i] = self.V[:,:,i-1] - self.dt*self.V[:,:,i-1]/self.tau +\
                            self.I[:,:,i-1]/(self.tau*self.g_l)
            #Enforce refractory period
            self.V[:,:,i] = self.V[:,:,i] - self.V[:,:,i]*self.R[:,:,i+self.ref_steps]

class LIF(Neuron):

    def __init__(self,  T, dt, tau_ref, J, trials=1, tau=0.02, g_l=50, thr=20, dtype=np.float32):

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


# class HodgkinHuxley(Neuron):
#
#     def __init__(self, t, trials=1, c_m=1.0, g_L=0.3, g_Na=120.0, g_K=36.0, E_Na=50.0, E_K=-77.0, E_L=-54.387, input=None, dtype=np.float32):
#
#         super(HodgkinHuxley, self).__init__(t, input, trials, dtype)
#
#         """
#
#         Hodgkin-Huxley neuron model
#
#         Parameters
#         ----------
#
#         t: 1D ndarray
#             A 1-dimensional numpy array containing time steps
#         trials : **Currently not implemented
#         c_m : float, optional
#             Capacitance of the membrane per unit area in [uF/cm^2]
#         g_L : float, optional
#             Leak conductance per unit area in [mS/cm^2]
#         g_K : float, optional
#             Potassium conductance per unit area in [mS/cm^2]
#         g_Na : float, optional
#             Sodium conductance per unit area in [mS/cm^2]
#         E_L : float, optional
#             Leak reversal potential [mV]
#         E_K : float, optional
#             Potassium reversal potential [mV]
#         E_Na : float, optional
#             Sodium reversal potential [mV]
#         input: ndarray, optional
#             Input current per unit area.
#             For examination of one or more neurons response to known input current(s)
#             (defaults to None)
#         dtype : numpy data type, optional
#             Data type to use for neuron state variables
#
#         """
#
#         self.c_m = c_m
#         self.g_Na = g_Na
#         self.g_K = g_K
#         self.g_L = g_L
#         self.E_Na = E_Na
#         self.E_K = E_K
#         self.E_L = E_L
#
#     def alpha_m(self, V):
#         """Channel gating kinetics. Functions of membrane voltage"""
#         return 0.1*(V+40.0)/(1.0 - sp.exp(-(V+40.0) / 10.0))
#
#     def beta_m(self, V):
#         """Channel gating kinetics. Functions of membrane voltage"""
#         return 4.0*sp.exp(-(V+65.0) / 18.0)
#
#     def alpha_h(self, V):
#         """Channel gating kinetics. Functions of membrane voltage"""
#         return 0.07*sp.exp(-(V+65.0) / 20.0)
#
#     def beta_h(self, V):
#         """Channel gating kinetics. Functions of membrane voltage"""
#         return 1.0/(1.0 + sp.exp(-(V+35.0) / 10.0))
#
#     def alpha_n(self, V):
#         """Channel gating kinetics. Functions of membrane voltage"""
#         return 0.01*(V+55.0)/(1.0 - sp.exp(-(V+55.0) / 10.0))
#
#     def beta_n(self, V):
#         """Channel gating kinetics. Functions of membrane voltage"""
#         return 0.125*sp.exp(-(V+65) / 80.0)
#
#     def I_Na(self, V, m, h):
#         return self.g_Na * m**3 * h * (V - self.E_Na)
#
#     def I_K(self, V, n):
#         return self.g_K  * n**4 * (V - self.E_K)
#
#     def I_L(self, V):
#         return self.g_L * (V - self.E_L)
#
#     def zero_state(self):
#
#         #Initialize state variables
#         self.V = np.zeros_like(self.input, dtype=self.dtype)
#         self.m = np.zeros_like(self.input, dtype=self.dtype)
#         self.n = np.zeros_like(self.input, dtype=self.dtype)
#         self.h = np.zeros_like(self.input, dtype=self.dtype)
#
#         self.V[0] = -65
#         self.m[0] = 0.05
#         self.h[0] = 0.6
#         self.n[0] = 0.32
#
#     @staticmethod
#     def dALLdt(X, t, self):
#
#         V, m, h, n = X
#
#         dVdt = (self.I_inj(t) - self.I_Na(V, m, h) - self.I_K(V, n) - self.I_L(V)) / self.c_m
#         dmdt = self.alpha_m(V)*(1.0-m) - self.beta_m(V)*m
#         dhdt = self.alpha_h(V)*(1.0-h) - self.beta_h(V)*h
#         dndt = self.alpha_n(V)*(1.0-n) - self.beta_n(V)*n
#         return dVdt, dmdt, dhdt, dndt
#
#     def call(self):
#
#         self.Zero_state()
#
#         """
#         Integrate neuron model
#         """
#
#         for i,t in enumerate(self.t[self.tau_ref:]):
#
#             #check if the neuron spiked in the last tau_ref time steps
#             self.R[:,:,i] = np.sum(self.Z[:,:,i-self.tau_ref:i], axis=-1)
#
#             #set inpu
# class HodgkinHuxley(Neuron):
#
#     def __init__(self, t, trials=1, c_m=1.0, g_L=0.3, g_Na=120.0, g_K=36.0, E_Na=50.0, E_K=-77.0, E_L=-54.387, input=None, dtype=np.float32):
#
#         super(HodgkinHuxley, self).__init__(t, input, trials, dtype)
#
#         """
#
#         Hodgkin-Huxley neuron model
#
#         Parameters
#         ----------
#
#         t: 1D ndarray
#             A 1-dimensional numpy array containing time steps
#         trials : **Currently not implemented
#         c_m : float, optional
#             Capacitance of the membrane per unit area in [uF/cm^2]
#         g_L : float, optional
#             Leak conductance per unit area in [mS/cm^2]
#         g_K : float, optional
#             Potassium conductance per unit area in [mS/cm^2]
#         g_Na : float, optional
#             Sodium conductance per unit area in [mS/cm^2]
#         E_L : float, optional
#             Leak reversal potential [mV]
#         E_K : float, optional
#             Potassium reversal potential [mV]
#         E_Na : float, optional
#             Sodium reversal potential [mV]
#         input: ndarray, optional
#             Input current per unit area.
#             For examination of one or more neurons response to known input current(s)
#             (defaults to None)
#         dtype : numpy data type, optional
#             Data type to use for neuron state variables
#
#         """
#
#         self.c_m = c_m
#         self.g_Na = g_Na
#         self.g_K = g_K
#         self.g_L = g_L
#         self.E_Na = E_Na
#         self.E_K = E_K
#         self.E_L = E_L
#
#     def alpha_m(self, V):
#         """Channel gating kinetics. Functions of membrane voltage"""
#         return 0.1*(V+40.0)/(1.0 - sp.exp(-(V+40.0) / 10.0))
#
#     def beta_m(self, V):
#         """Channel gating kinetics. Functions of membrane voltage"""
#         return 4.0*sp.exp(-(V+65.0) / 18.0)
#
#     def alpha_h(self, V):
#         """Channel gating kinetics. Functions of membrane voltage"""
#         return 0.07*sp.exp(-(V+65.0) / 20.0)
#
#     def beta_h(self, V):
#         """Channel gating kinetics. Functions of membrane voltage"""
#         return 1.0/(1.0 + sp.exp(-(V+35.0) / 10.0))
#
#     def alpha_n(self, V):
#         """Channel gating kinetics. Functions of membrane voltage"""
#         return 0.01*(V+55.0)/(1.0 - sp.exp(-(V+55.0) / 10.0))
#
#     def beta_n(self, V):
#         """Channel gating kinetics. Functions of membrane voltage"""
#         return 0.125*sp.exp(-(V+65) / 80.0)
#
#     def I_Na(self, V, m, h):
#         return self.g_Na * m**3 * h * (V - self.E_Na)
#
#     def I_K(self, V, n):
#         return self.g_K  * n**4 * (V - self.E_K)
#
#     def I_L(self, V):
#         return self.g_L * (V - self.E_L)
#
#     def zero_state(self):
#
#         #Initialize state variables
#         self.V = np.zeros_like(self.input, dtype=self.dtype)
#         self.m = np.zeros_like(self.input, dtype=self.dtype)
#         self.n = np.zeros_like(self.input, dtype=self.dtype)
#         self.h = np.zeros_like(self.input, dtype=self.dtype)
#
#         self.V[0] = -65
#         self.m[0] = 0.05
#         self.h[0] = 0.6
#         self.n[0] = 0.32
#
#     @staticmethod
#     def dALLdt(X, t, self):
#
#         V, m, h, n = X
#
#         dVdt = (self.I_inj(t) - self.I_Na(V, m, h) - self.I_K(V, n) - self.I_L(V)) / self.c_m
#         dmdt = self.alpha_m(V)*(1.0-m) - self.beta_m(V)*m
#         dhdt = self.alpha_h(V)*(1.0-h) - self.beta_h(V)*h
#         dndt = self.alpha_n(V)*(1.0-n) - self.beta_n(V)*n
#         return dVdt, dmdt, dhdt, dndt
#
#     def call(self):
#
#         self.Zero_state()
#
#         """
#         Integrate neuron model
#         """
#
#         for i,t in enumerate(self.t[self.tau_ref:]):
#
#             #check if the neuron spiked in the last tau_ref time steps
#             self.R[:,:,i] = np.sum(self.Z[:,:,i-self.tau_ref:i], axis=-1)
#
#             #set input current
#             if self.input is None:
#                 pass
#             else:
#                 i_in = (self.input[:,:,i]-self.I_Na(self.V[:,:,i], self.m, self.h)-\
#                 self.I_K(self.V[:,:,i], n)-self.I_L(self.V[:,:,i]))/self.c_m
#
#             self.V[:,:,i] = self.V[:,:,i-1] + self.dt*x
#             self.V = self.V - self.V*self.R
#
#             #apply spike function to current time step
#             self.Z[:,:,i] = self.spike_function(self.V[:,:,i])
#
#     def plot(self):
#
#         plt.figure()
#
#         plt.subplot(4,1,1)
#         plt.title('Hodgkin-Huxley Neuron')
#         plt.plot(self.t, self.V, 'k')
#         plt.ylabel('V (mV)')
#
#         plt.subplot(4,1,2)
#         plt.plot(self.t, self.ina, 'c', label='$I_{Na}$')
#         plt.plot(self.t, self.ik, 'y', label='$I_{K}$')
#         plt.plot(self.t, self.il, 'm', label='$I_{L}$')
#         plt.ylabel('Current')
#         plt.legend()
#
#         plt.subplot(4,1,3)
#         plt.plot(self.t, self.m, 'r', label='m')
#         plt.plot(self.t, self.h, 'g', label='h')
#         plt.plot(self.t, self.n, 'b', label='n')
#         plt.ylabel('Gating Value')
#         plt.legend()
#
#         plt.subplot(4,1,4)
#         plt.plot(self.t, self.input, 'k')
#         plt.xlabel('t (ms)')
#         plt.ylabel('$I_{inj}$ ($\\mu{A}/cm^2$)')
#         plt.ylim(-1, 40)
#
#         plt.tight_layout()

#             if self.input is None:
#                 pass
#             else:
#                 i_in = (self.input[:,:,i]-self.I_Na(self.V[:,:,i], self.m, self.h)-\
#                 self.I_K(self.V[:,:,i], n)-self.I_L(self.V[:,:,i]))/self.c_m
#
#             self.V[:,:,i] = self.V[:,:,i-1] + self.dt*x
#             self.V = self.V - self.V*self.R
#
#             #apply spike function to current time step
#             self.Z[:,:,i] = self.spike_function(self.V[:,:,i])
#
#     def plot(self):
#
#         plt.figure()
#
#         plt.subplot(4,1,1)
#         plt.title('Hodgkin-Huxley Neuron')
#         plt.plot(self.t, self.V, 'k')
#         plt.ylabel('V (mV)')
#
#         plt.subplot(4,1,2)
#         plt.plot(self.t, self.ina, 'c', label='$I_{Na}$')
#         plt.plot(self.t, self.ik, 'y', label='$I_{K}$')
#         plt.plot(self.t, self.il, 'm', label='$I_{L}$')
#         plt.ylabel('Current')
#         plt.legend()
#
#         plt.subplot(4,1,3)
#         plt.plot(self.t, self.m, 'r', label='m')
#         plt.plot(self.t, self.h, 'g', label='h')
#         plt.plot(self.t, self.n, 'b', label='n')
#         plt.ylabel('Gating Value')
#         plt.legend()
#
#         plt.subplot(4,1,4)
#         plt.plot(self.t, self.input, 'k')
#         plt.xlabel('t (ms)')
#         plt.ylabel('$I_{inj}$ ($\\mu{A}/cm^2$)')
#         plt.ylim(-1, 40)
#
#         plt.tight_layout()
#         plt.show()
