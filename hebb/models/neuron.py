import numpy as np
from .network import *
import scipy as sp
import pylab as plt
from scipy.integrate import odeint

class LIF():

    def __init__(self, t, tau=20., thr=0.615, tau_ref=3, batches=1, dtype=np.float32):

        #Basic parameters
        self.dt = np.mean(np.diff(t))
        self.t = t
        self.dtype = dtype
        self.tau = tau
        self.tau_ref = tau_ref
        self.thr = thr
        self.batches = batches

        self.C_m  =   1.0
        """membrane capacitance, in uF/cm^2"""

        self.g_L  =   0.3
        """Leak maximum conductances, in mS/cm^2"""

    def spike_function(self, v):

        """
        Thresholds the voltage vector to update the observable state tensor
        """

        z = (v > self.thr).astype('int')

        return z


    def zero_state(self):

        #pad along time axis for calculating the refractory variable as a sum over z
        self.v = np.zeros_like(self.input, dtype=self.dtype)
        self.z = np.zeros_like(self.input, dtype=self.dtype)
        self.r = np.zeros_like(self.input, dtype=self.dtype)

    def call(self, input, plot=False):

        self.input = input
        if len(self.input.shape) == 1:
            self.input = self.input.reshape((1,) + self.input.shape)
            self.input = np.repeat(self.input, self.batches, axis=0)
        self.zero_state()

        for i,t in enumerate(self.t[self.tau_ref:]):

            #check if the neuron spiked in the last tau_ref time steps
            self.r[:,i] = np.sum(self.z[:,i-self.tau_ref:i], axis=-1)

            #set input current
            i_in = self.input[:,i]

            self.v[:,i] = self.v[:,i-1] + self.dt*(-self.g_L*self.v[:,i-1] + i_in)/self.C_m
            self.v = self.v - self.v*self.r

            #apply spike function to current time step
            self.z[:,i] = self.spike_function(self.v[:,i])

        if plot:

            fig, ax = plt.subplots(4,1, sharex=True)

            ax[0].plot(self.t, self.v[0,:], 'k')
            ax[0].hlines(self.thr, self.t.min(), self.t.max(), color='red')
            ax[0].hlines(0, self.t.min(), self.t.max(), color='blue')
            ax[0].set_ylabel('V (mV)')

            ax[1].plot(self.t, self.input[0,:], 'k')
            ax[1].set_xlabel('t (ms)')
            ax[1].set_ylabel('$I_{inj}$(t) ($\\mu{A}/cm^2$)')

            ax[2].plot(self.t, self.r[0,:], 'k')
            ax[2].set_xlabel('t (ms)')
            ax[2].set_ylabel('$R(t)$')

            ax[3].plot(self.t, self.z[0,:], 'k')
            ax[3].set_xlabel('t (ms)')
            ax[3].set_ylabel('$z(t)$')

            plt.tight_layout()
            plt.show()


class HodgkinHuxley():

    C_m  =   1.0
    """membrane capacitance, in uF/cm^2"""

    g_Na = 120.0
    """Sodium (Na) maximum conductances, in mS/cm^2"""

    g_K  =  36.0
    """Postassium (K) maximum conductances, in mS/cm^2"""

    g_L  =   0.3
    """Leak maximum conductances, in mS/cm^2"""

    E_Na =  50.0
    """Sodium (Na) Nernst reversal potentials, in mV"""

    E_K  = -77.0
    """Postassium (K) Nernst reversal potentials, in mV"""

    E_L  = -54.387
    """Leak Nernst reversal potentials, in mV"""

    t = sp.arange(0.0, 450.0, 0.01)
    """ The time to integrate over """

    def alpha_m(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 0.1*(V+40.0)/(1.0 - sp.exp(-(V+40.0) / 10.0))

    def beta_m(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 4.0*sp.exp(-(V+65.0) / 18.0)

    def alpha_h(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 0.07*sp.exp(-(V+65.0) / 20.0)

    def beta_h(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 1.0/(1.0 + sp.exp(-(V+35.0) / 10.0))

    def alpha_n(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 0.01*(V+55.0)/(1.0 - sp.exp(-(V+55.0) / 10.0))

    def beta_n(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 0.125*sp.exp(-(V+65) / 80.0)

    def I_Na(self, V, m, h):
        return self.g_Na * m**3 * h * (V - self.E_Na)

    def I_K(self, V, n):
        return self.g_K  * n**4 * (V - self.E_K)

    def I_L(self, V):
        return self.g_L * (V - self.E_L)

    def I_inj(self, t):
        return 10*(t>100) - 10*(t>200) + 35*(t>300) - 35*(t>400)

    @staticmethod
    def dALLdt(X, t, self):

        V, m, h, n = X

        dVdt = (self.I_inj(t) - self.I_Na(V, m, h) - self.I_K(V, n) - self.I_L(V)) / self.C_m
        dmdt = self.alpha_m(V)*(1.0-m) - self.beta_m(V)*m
        dhdt = self.alpha_h(V)*(1.0-h) - self.beta_h(V)*h
        dndt = self.alpha_n(V)*(1.0-n) - self.beta_n(V)*n
        return dVdt, dmdt, dhdt, dndt

    def main(self):

        X = odeint(self.dALLdt, [-65, 0.05, 0.6, 0.32], self.t, args=(self,))
        V = X[:,0]
        m = X[:,1]
        h = X[:,2]
        n = X[:,3]
        ina = self.I_Na(V, m, h)
        ik = self.I_K(V, n)
        il = self.I_L(V)

        plt.figure()

        plt.subplot(4,1,1)
        plt.title('Hodgkin-Huxley Neuron')
        plt.plot(self.t, V, 'k')
        plt.ylabel('V (mV)')

        plt.subplot(4,1,2)
        plt.plot(self.t, ina, 'c', label='$I_{Na}$')
        plt.plot(self.t, ik, 'y', label='$I_{K}$')
        plt.plot(self.t, il, 'm', label='$I_{L}$')
        plt.ylabel('Current')
        plt.legend()

        plt.subplot(4,1,3)
        plt.plot(self.t, m, 'r', label='m')
        plt.plot(self.t, h, 'g', label='h')
        plt.plot(self.t, n, 'b', label='n')
        plt.ylabel('Gating Value')
        plt.legend()

        plt.subplot(4,1,4)
        i_inj_values = [self.I_inj(t) for t in self.t]
        plt.plot(self.t, i_inj_values, 'k')
        plt.xlabel('t (ms)')
        plt.ylabel('$I_{inj}$ ($\\mu{A}/cm^2$)')
        plt.ylim(-1, 40)

        plt.tight_layout()
        plt.show()
