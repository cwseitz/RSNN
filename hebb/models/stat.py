import numpy as np
from copy import deepcopy
from scipy.stats import norm

################################################################################
##
##      Statistical models that can be used for comparisons with
##      real network simulations for arbitrary parameterizations
##
################################################################################

################################################################################
## Author: Clayton Seitz
## Copyright: 2021, The Hebb Project
## Email: cwseitz@uchicago.edu
################################################################################

class Brownian:

    def __init__(self, t, V0, sigma, batch_size=1, dtype=np.float32):

        """

        Integrate a Langevin equation when the diffusion term is a Gaussian white noise
        and the drift term is zero (a Wiener Process)

        """

        self.t = t
        self.V0 = V0
        self.nsteps = len(t)
        self.dt = np.mean(np.diff(t))
        self.sigma = sigma
        self.batch_size = batch_size
        self.V = []

    def forward(self):

        """
        Generate an instance of Brownian motion (i.e. the Wiener process):

            X(t) = X(0) + N(0, sigma**2 * t; 0, t)

        where N(a,b; t0, t1) is a normally distributed random variable with mean a and
        variance b.  The parameters t0 and t1 make explicit the statistical
        independence of N on different time intervals; that is, if [t0, t1) and
        [t2, t3) are disjoint intervals, then N(a, b; t0, t1) and N(a, b; t2, t3)
        are independent.

        Written as an iteration scheme,

            X(t + dt) = X(t) + N(0, sigma**2 * dt; t, t+dt)


        If `x0` is an array (or array-like), each value in `x0` is treated as
        an initial condition, and the value returned is a numpy array with one
        more dimension than `x0`.

        Arguments
        ---------
        x0 : float or numpy array (or something that can be converted to a numpy array
             using numpy.asarray(x0)).
            The initial condition(s) (i.e. position(s)) of the Brownian motion.
        nsteps : int
            The number of steps to take.
        dt : float
            The time step.
        sigma : float
            sigma determines the "speed" of the Brownian motion.  The random variable
            of the position at time t, X(t), has a normal distribution whose mean is
            the position at time t=0 and whose variance is sigma**2*t.

        Returns
        -------
        """

        self.dV = []

        for i in range(self.batch_size):

            # For each element of x0, generate a sample of n numbers from a normal distribution
            self.dV_i = norm.rvs(size=(self.nsteps,), scale=self.sigma*np.sqrt(self.dt))
            self.dV.append(self.dV_i)

        self.dV = np.array(self.dV).T #rows are time, columns are simulations
        self.V = np.cumsum(self.dV, axis=0) #cumsum over rows
        self.V += self.V0

        return self.V


class StationaryOU:

    def __init__(self, t, tau, sigma, dv=0.001, batch_size=1, v_max=1, V_R=-1, dtype=np.float32):

        """

        Integrate a Langevin equation for stationary white noise
        (an Ornstein Uhlenbeck Process)

        Parameters
        ----------

        nsteps: float
            Number of time steps before the simulation terminates
        V0 : float
            Initial condition for the stochastic variable V
        alpha: float
            Rate parameter for the linear drift of the white noise process
            (see equation above)
        sigma: float
            Noise amplitude
        batch_size: int
            Number of simulations to run
        v_min : int
            Minimum value for the voltage domain
        v_max : int
            Maximum value for the voltage domain
        dv : float
            Resolution for the voltage domain

        """

        #Params
        self.nsteps = len(t)
        self.dt = np.mean(np.diff(t))
        self.V_R = V_R
        self.alpha = 1/tau
        self.sigma = sigma
        self.batch_size = batch_size
        self.v_max = v_max
        self.dv = dv
        self.n_v = int(round(2*self.v_max/self.dv))
        self._V = np.linspace(-self.v_max, self.v_max, self.n_v)

        #Arrays for simulation history
        self.V = np.zeros((self.nsteps, self.batch_size))
        self.P_S = np.zeros((self.n_v, self.nsteps)); self.P_S[0,:] = 1
        self.P_A = deepcopy(self.P_S)
        self.P_N = deepcopy(self.P_S)

    def solve_fp_analytic(self):

        for n in range(1, self.nsteps):
            var = (self.sigma**2/(2*self.alpha))*(1-np.exp(-2*self.alpha*n*self.dt))
            mu = self.V_R*np.exp(-self.alpha*n*self.dt)
            P_t = np.sqrt(1/(2*np.pi*var))*np.exp(-((self._V-mu)**2)/(2*var))
            self.P_A[:,n] = P_t
        return self.P_A

    def histogram(self):

        for i in range(self.nsteps):
            vals, bins = np.histogram(self.V[i,:], bins=self.n_v, range=(-self.v_max,self.v_max), density=False)
            self.P_S[:,i] = vals/(np.sum(vals)*self.dv)


    def forward(self):

        self.V[0,:] = self.V_R
        noise = np.random.normal(loc=0.0,scale=1.0,size=(self.nsteps,self.batch_size))*np.sqrt(self.dt) #define noise process
        for i in range(1,self.nsteps):
            for j in range(self.batch_size):
                self.V[i,j] = self.V[i-1,j] - self.dt*self.alpha*(self.V[i-1,j]) + self.sigma*noise[i,j]

class NonStationaryOU:

    def __init__(self, nsteps, V_R, tau, mu, sigma, dt=0.001, batch_size=1, xmin=0, xmax=1, dtype=np.float32):

        """

        Monte-Carlo integration of a Langevin equation for
        non-stationary white noise (an Ornstein Uhlenbeck Process)

        Parameters
        ----------

        nsteps: float
            Number of time steps before the simulation terminates
        V0 : float
            Initial condition for the stochastic variable V
        alpha: float
            Rate parameter for the linear drift of the white noise process
            (see equation above)
        mu: ndarray
            The mean of the non-stationary white noise as a function of time
        sigma: float
            Noise amplitude
        batch_size: int
            Number of simulations to run

        """

        #Params
        self.nsteps = nsteps
        self.dt = dt
        self.V_R = V_R
        self.alpha = 1/tau
        self.mu = mu
        self.sigma = sigma
        self.batch_size = batch_size
        self.xmin = xmin
        self.xmax = xmax

        #Arrays for simulation history
        self.V = np.zeros((self.nsteps, self.batch_size))
        self.P = []

    def forward(self):

        self.V[0,:] = self.V_R
        noise = np.random.normal(loc=0.0,scale=1.0,size=(self.nsteps,self.batch_size))*np.sqrt(self.dt) #define noise process
        for i in range(1,self.nsteps):
            for j in range(self.batch_size):
                self.V[i,j] = self.V[i-1,j] - self.dt*self.alpha*(self.V[i-1,j]) + self.mu[i] + self.sigma*noise[i,j]

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

        return self.spikes

    def to_currents(self, J):
        self.currents = np.einsum('ij,jhk->ihk', J, self.spikes)
        return self.currents

class CompoundPoisson:

    def __init__(self, T, dt, J, trials=1, rates=None, dtype=np.float32):

        """

        Generate a compound Poisson process

        Parameters
        ----------

        T: float
            Total simulation time in seconds
        dt: float
            Time resolution in seconds
        J : ndarray,
            A vector of weights representing the weight of an event.
            Requires a shape (N,)
        trials : int
            Number of stimulations to run
        rates: ndarray,
            A tensor of rates where rows are units and columns are time
            Has shape (N,trials,nsteps) and should be constant on axis=1
        dtype : numpy data type
            Data type to use for neuron state variables

        """

        self.T = T
        self.dt = dt
        self.N = J.shape[0]
        self.J = J
        self.nsteps = 1 + int(round(T/dt))
        self.trials = trials

        self.rates = rates
        if self.rates is None:
            self.r0 = 20 #default rate (Hz)
            self.rates = self.r0*np.ones((self.N, self.trials, self.nsteps))

    def run_generator(self):

        self.r = self.rates*self.dt
        self.x = np.random.uniform(0,1,size=(self.N,self.trials,self.nsteps))
        self.spikes = np.array(self.x < self.r, dtype=np.int32)

        #sum over the first axis, weighting by J
        self.current = np.einsum('i,ijk->jk', self.J, self.spikes)
        return self.current
