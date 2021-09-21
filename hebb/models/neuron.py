import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.integrate import odeint
from .network import *

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
##     State variables are 3D tensors with shape (N, batches, nsteps)
##
################################################################################

################################################################################
## Author: Clayton Seitz
## Copyright: 2021, The Hebb Project
## Email: cwseitz@uchicago.edu
################################################################################

class Neuron:

    def __init__(self, t, N=1, batches=1, X=None, input=None, dtype=np.float32):

        """

        Neuron base class

        Parameters
        ----------

        t: 1D ndarray
            A 1-dimensional numpy array containing time steps
        X : ndarray
            Input spikes
        N: int
            Number of neurons to simulate
        batches : int
            Number of stimulations to run
        input: ndarray, optional
            Input current per unit area.
            For examination of one or more neurons response to known input current(s)
            (defaults to None)

        """

        #Make sure either input spikes or input currents are specified
        if X is None and input is None:
            raise ValueError('Neither input currents nor input spikes were specified')

        #Basic parameters
        self.dt = np.mean(np.diff(t))
        self.t = t
        self.nsteps = len(self.t)
        self.N = N
        self.dtype = dtype
        self.input = input
        self.X = X
        self.batches = batches

        if input is None:
            self.connect()
        else:
            if self.input.shape != (self.N, self.batches, self.nsteps):
                raise ValueError('Input shape is not (N, batches, nsteps)')

    def connect(self):

        """
        Generate network connectivity
        """

        self.J = np.random.normal(0,1,size=(self.N, self.N))
        self.W = np.random.normal(0,1,size=(self.N,self.X.shape[0]))


class LIF(Neuron):

    def __init__(self, t, N=1, batches=1, X=None, input=None, c_m=1.0, g_L=0.3, thr=0.615, tau_ref=3, dtype=np.float32):

        super(LIF, self).__init__(t, N=N, batches=batches, X=X, input=input, dtype=dtype)

        """

        Leaky Integrate & Fire (LIF) neuron model

        Parameters
        ----------

        t: 1D ndarray
            A 1-dimensional numpy array containing time steps
        N: int
            Number of neurons to simulate
        batches : int
            Number of stimulations to run
        c_m : float
            Capacitance of the membrane per unit area in [uF/cm^2]
        g_L : float
            Leak conductance per unit area in [mS/cm^2]
        thr : float
            Firing threshold
        tau_ref : int
            An integer n s.t. refactory time is equal to n*dt
        input: ndarray, optional
            Input current per unit area. For examination of one or more neurons
            response to known input current(s) (defaults to None).
        dtype : numpy data type
            Data type to use for neuron state variables

        """

        #LIF specific parameters

        self.g_L = g_L
        self.c_m = c_m
        self.tau_ref = tau_ref
        self.thr = thr

    def spike_function(self, v):
        z = (v > self.thr).astype('int')
        return z

    def zero_state(self):

        #Initialize state variables
        self.I = np.zeros(shape=(self.N, self.batches, self.nsteps+self.tau_ref), dtype=self.dtype)
        self.V = np.zeros(shape=(self.N, self.batches, self.nsteps+self.tau_ref), dtype=self.dtype)
        self.Z = np.zeros(shape=(self.N, self.batches, self.nsteps+self.tau_ref), dtype=np.int8)
        self.R = np.zeros(shape=(self.N, self.batches, self.nsteps+self.tau_ref), dtype=np.int8)

        if self.input is None:
            pass
        else:
            self.I[:,:,self.tau_ref:] = self.input

    def call(self):

        self.zero_state()

        for i in range(self.tau_ref-1, self.nsteps+self.tau_ref):
            #check if the neuron spiked in the last tau_ref time steps
            self.R[:,:,i] = np.sum(self.Z[:,:,i-self.tau_ref:i], axis=-1)

            #set input current if spikes were provided
            if self.input is None:
                i_in = np.matmul(self.W, self.X[:,:,i-1-self.tau_ref])
                i_re = np.matmul(self.J, self.Z[:,:,i-1])
                self.I[:,:,i] =  i_in + i_re

            self.V[:,:,i] = self.V[:,:,i-1] + self.dt*(-self.g_L*self.V[:,:,i-1] +\
                            self.I[:,:,i])/self.c_m
            self.V = self.V - self.V*self.R

            #apply spike function to current time step
            self.Z[:,:,i] = self.spike_function(self.V[:,:,i])

    def plot_weights(self):

        fig, ax = plt.subplots(1,2)
        ax[0].imshow(self.J, cmap='gray')
        ax[1].imshow(self.W, cmap='gray')
        plt.tight_layout()

    def plot_activity(self, batch=0):

        fig, ax = plt.subplots(4,1, sharex=True)

        nu_n = np.sum(self.Z[:,batch,self.tau_ref:], axis=0)/self.N
        nu_x = np.sum(self.X[:,batch,:], axis=0)/self.X.shape[0]

        ax[0].imshow(self.V[:,batch,self.tau_ref:], cmap='gray')
        ax[1].imshow(self.Z[:,batch,self.tau_ref:], cmap='gray')
        ax[2].imshow(self.X[:,batch,:], cmap='gray')
        ax[0].set_ylabel('N')
        ax[1].set_ylabel('N')
        ax[2].set_ylabel('X')
        ax[3].set_xlabel('Time')
        ax[3].plot(nu_n, color='red', label='Primary')
        ax[3].plot(nu_x, color='blue', label='Input')
        ax[3].set_ylabel('A(t)')
        plt.legend()

    def plot_input_stats(self, bins=10):

        fig, ax = plt.subplots(4,1)

        mu = np.mean(self.I, axis=(0,1))
        var = np.std(self.I, axis=(0,1))**2

        ax[0].plot(mu, color='red')
        ax[1].plot(var, color='blue')
        ax[2].hist(mu, bins=100, color='red')
        ax[3].hist(var, bins=100, color='blue')
        plt.tight_layout()

    def plot_voltage_stats(self, bins=10):

        fig, ax = plt.subplots()
        colormap = cm.get_cmap('coolwarm')
        colors = colormap(np.linspace(0, 1, self.nsteps))

        #compute the histogram of values over (unit, batch) matrix
        hist_arr, edges_arr = [], []
        for t in range(self.nsteps):
            hist, edges = np.histogram(self.V[:,:,t], bins=bins, density=True)
            ax.plot(edges[:-1], hist, color=colors[t], alpha=0.5)

        ax.set_xlabel('Voltage (a.u.)')
        ax.set_ylabel('PDF')
        plt.tight_layout()

    def plot_unit(self, unit=0, batch=0):

        #Plot input and state variables for a single unit in a single batch
        fig, ax = plt.subplots(4,1, sharex=True)

        ax[0].plot(self.t, self.V[unit,batch,:self.nsteps], 'k')
        ax[0].hlines(self.thr, self.t.min(), self.t.max(), color='red')
        ax[0].hlines(0, self.t.min(), self.t.max(), color='blue')
        ax[0].set_ylabel('V (mV)')

        ax[1].plot(self.t, self.I[unit,batch,self.tau_ref:], 'k')
        ax[1].set_xlabel('t (ms)')
        ax[1].set_ylabel('$I$(t) ($\\mu{A}/cm^2$)')

        ax[2].plot(self.t, self.R[unit,batch,:self.nsteps], 'k')
        ax[2].set_xlabel('t (ms)')
        ax[2].set_ylabel('$R(t)$')

        ax[3].plot(self.t, self.Z[unit,batch,:self.nsteps], 'k')
        ax[3].set_xlabel('t (ms)')
        ax[3].set_ylabel('$Z(t)$')

        plt.tight_layout()


# class HodgkinHuxley(Neuron):
#
#     def __init__(self, t, batches=1, c_m=1.0, g_L=0.3, g_Na=120.0, g_K=36.0, E_Na=50.0, E_K=-77.0, E_L=-54.387, input=None, dtype=np.float32):
#
#         super(HodgkinHuxley, self).__init__(t, input, batches, dtype)
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
#         batches : **Currently not implemented
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
#         plt.show()
