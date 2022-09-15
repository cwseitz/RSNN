import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from copy import deepcopy
from matplotlib import cm

##################################################
## Library of statistical models
##################################################
## Author: Clayton Seitz
## Copyright: 2021, The Hebb Project
## Email: cwseitz@uchicago.edu
##################################################

class OrnsteinUhlenbeck:

    def __init__(self, T, dt, tau, sigma, dx=0.1, x_max=1, x0=0, trials=1000, dtype=np.float32):

        """

        Integrate a Langevin equation for stationary white noise
        (an Ornstein Uhlenbeck Process)

        Parameters
        ----------
        """

        #Params
        self.nsteps = int(round(T/dt))
        self.dt = dt
        self.dx = dx
        self.x0 = x0
        self.x_max = x_max
        self.nx = int(round(2*self.x_max/self.dx))
        self.alpha = 1/tau
        self.sigma = sigma
        self.trials = trials
        self._x = np.linspace(-self.x_max, self.x_max, self.nx)

        #Arrays for simulation history
        self.X = np.zeros((self.nsteps, self.trials))
        self.p1 = np.zeros((self.nx, self.nsteps)); self.p1[0,:] = 1
        self.p2 = deepcopy(self.p1)

    def solve(self):

        for n in range(1, self.nsteps):
            var = (self.sigma**2/(2*self.alpha))*(1-np.exp(-2*self.alpha*n*self.dt))
            mu = self.x0*np.exp(-self.alpha*n*self.dt)
            p0 = np.sqrt(1/(2*np.pi*var))*np.exp(-((self._x-mu)**2)/(2*var))
            self.p2[:,n] = p0
        return self.p2

    def histogram(self):

        for i in range(self.nsteps):
            vals, bins = np.histogram(self.X[i,:], bins=self.nx, range=(-self.x_max,self.x_max), density=False)
            self.p1[:,i] = vals/(np.sum(vals)*self.dx)

    def forward(self):

        self.X[0,:] = self.x0
        noise = np.random.normal(loc=0.0,scale=1.0,size=(self.nsteps,self.trials))*np.sqrt(self.dt) #define noise process
        for i in range(1,self.nsteps):
            for j in range(self.trials):
                self.X[i,j] = self.X[i-1,j] - self.dt*self.alpha*(self.X[i-1,j]) + self.sigma*noise[i,j]

class OrnsteinUhlenbeckNS:

    def __init__(self, T, dt, tau, stim, sigma, x0=0, trials=1, xmin=0, xmax=1, dtype=np.float32):

        """

        Monte-Carlo simulation of a Langevin equation for
        non-stationary white noise (an Ornstein Uhlenbeck Process)

        Parameters
        ----------

        T: float
            Duration of the simulation in seconds
        V0 : float
            Initial condition for the stochastic variable V
        alpha: float
            Rate parameter for the linear drift of the white noise process
            (see equation above)
        stim: ndarray
            The stimuluss
        sigma: float
            Noise amplitude
        trials: int
            Number of simulations to run

        """

        #Params
        self.nsteps = int(round(T/dt))
        self.dt = dt
        self.alpha = 1/tau
        self.stim = stim
        self.sigma = sigma
        self.trials = trials
        self.x0 = x0
        self.xmin = xmin
        self.xmax = xmax

        #Arrays for simulation history
        self.X = np.zeros((self.nsteps, self.trials))
        self.P = []

    def forward(self):

        noise = np.random.normal(loc=0.0,scale=1.0,size=(self.nsteps,self.trials))*np.sqrt(self.dt) #define noise process
        for i in range(1,self.nsteps):
            for j in range(self.trials):
                self.X[i,j] = self.X[i-1,j] - self.dt*self.alpha*(self.X[i-1,j]) + self.stim[i] + self.sigma*noise[i,j]

    def plot_trajectories(self):

        colormap = cm.get_cmap('viridis')
        colors = colormap(np.linspace(0, 1, self.trials))
        norm = mpl.colors.Normalize(vmin=0, vmax=self.trials)

        fig, ax = plt.subplots()

        for i in range(self.trials):
            ax.plot(self.X[:,i], color=colors[i], alpha=0.75)

        ax.set_xlabel('Time')
        ax.set_ylabel('X')

        plt.tight_layout()
        plt.grid()

class Poisson:

    """

    Generate an ensemble of Poisson spike trains from a vector of rates

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

    def __init__(self, T, dt, N, trials=1, rates=None, random_select=None):

        self.T = T
        self.dt = dt
        self.N = N
        self.nsteps = 1 + int(round(T/dt))
        self.random_select = random_select
        self.trials = trials

        if rates is None:
            self.r0 = 20 #default rate (Hz)
            rates = self.r0*np.ones((self.N, self.trials, self.nsteps))

        self.rates = rates
        self.run_generator()

    def run_generator(self):

        self.r = self.rates*self.dt
        self.x = np.random.uniform(0,1,size=(self.N,self.trials,self.nsteps))
        self.spikes = np.array(self.x < self.r, dtype=np.int32)

        if self.random_select != None:
            rng = default_rng()
            x = rng.choice(self.N, size=self.random_select, replace=False)
            self.spikes[x,:,:] = 0

    def to_currents(self, J):
        self.currents = np.einsum('ij,jhk->ihk', J, self.spikes)
        return self.currents
