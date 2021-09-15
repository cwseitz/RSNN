import numpy as np
from scipy.stats import norm

class Brownian:

    def __init__(self, V0, nsteps, dt, beta, batch_size=1, dtype=np.float32):

        """

        Integrate a Langevin equation when the diffusion term is a Gaussian white noise
        and the drift term is zero (a Wiener Process)

        """

        self.V0 = V0
        self.nsteps = nsteps
        self.dt = dt
        self.beta = beta
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
            self.dV_i = norm.rvs(size=(self.nsteps,), scale=self.beta*np.sqrt(self.dt))
            self.dV.append(self.dV_i)

        self.dV = np.array(self.dV).T #rows are time, columns are simulations
        self.V = np.cumsum(self.dV, axis=0) #cumsum over rows
        self.V += self.V0

        return self.V


class OrnsteinUhlenbeck:

    def __init__(self, nsteps, V0, alpha, beta, dt=0.001, batch_size=1, dtype=np.float32):

        """

        Integrate a Langevin equation when the noise term is a Gaussian white noise
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
        beta: float
            Noise amplitude
        batch_size: int
            Number of simulations to run

        """

        #Params
        self.nsteps = nsteps
        self.dt = dt
        self.V0 = V0
        self.alpha = alpha
        self.beta = beta
        self.batch_size = batch_size

        #Arrays for simulation history
        self.V = np.zeros((self.nsteps,batch_size))
        self.P = []

        #Generate an instance of Brownian motion
        self.b = Brownian(self.V0, self.nsteps, self.dt, self.beta, self.batch_size)
        self.b.forward()
        self.dW = self.b.dV

    def solve_analytic(self):

        self.dom = np.linspace(-2, 2, 100)
        for n in range(1, self.nsteps):
            var = (self.beta**2/(2*self.alpha))*(1-np.exp(-2*self.alpha*n*self.dt))
            mu = self.V0*np.exp(-self.alpha*n*self.dt)
            P_t = np.sqrt(1/np.sqrt(2*np.pi*var))*np.exp(-((self.dom-mu)**2)/(2*var))
            self.P.append(P_t)
        self.P = np.array(self.P)
        return self.P

    def forward(self):

        self.V += self.V0
        for t in range(1, self.nsteps):
            for b in range(self.batch_size):
                self.V[t,b] = (1-self.alpha)*self.V[t-1,b] + self.dW[t,b]
