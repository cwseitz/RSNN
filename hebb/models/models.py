import numpy as np

class LangevinGWN:

    def __init__(self, nsteps, V_0, J, alpha, tau=1, batch_size=1, dtype=np.float32):

        """

        Integrate a Langevin equation when the noise term is a Gaussian white noise

        tau*dV/dt = -V + J*s with s ~ N(alpha*t,1)

        Parameters
        ----------

        nsteps: float
            Number of time steps before the simulation terminates
        V_0 : float
            Initial condition for the stochastic variable V
        J: float
            Noise amplitude
        alpha: float
            Rate parameter for the linear drift of the white noise process
            (see equation above)
        tau: float
            The dissipation (leak) parameter (see equation above)
        batch_size: int
            Number of simulations to run

        """

        #Params
        self.nsteps = nsteps
        self.V_0 = V_0
        self.J = J
        self.alpha = alpha
        self.tau = tau
        self.batch_size = batch_size

        #Arrays for simulation history
        self.V = np.zeros((self.nsteps,batch_size))
        self.V[0] = self.V_0
        self.X = np.zeros((self.nsteps,batch_size))

    def forward(self):

        for t in range(1, self.nsteps):
            for b in range(self.batch_size):
                x = np.random.normal(self.alpha*t, 1)
                self.X[t,b] = x
                self.V[t,b] = (1/self.tau)*(-self.V[t-1,b] + self.J*x)
