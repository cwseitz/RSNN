import matplotlib.pyplot as plt
import numpy as np

class ExInLIF:

    def __init__(self, n_in, n_rec, p_xx, nsteps, p_e=0.8, tau=20.,
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
        self.tau = tau
        self.tau_ref = tau_ref
        self.decay = -dt/self.tau
        self.thr = thr
        self.nsteps = nsteps
        self.batches = batches

        #Network connectivity
        self.in_cmg = InputConnectivityGenerator(self.n_in,self. n_rec)
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
        self.v = np.zeros(shape=(self.n_rec, batches, self.nsteps+self.tau_ref), dtype=self.dtype)
        self.z = np.zeros(shape=(self.n_rec, batches, self.nsteps+self.tau_ref), dtype=np.int8)
        self.r = np.zeros(shape=(self.n_rec, batches, self.nsteps+self.tau_ref), dtype=np.int8)

    def call(self, input):

        for t in range(self.tau_ref, self.nsteps):

            #check if the neuron spiked in the last tau_ref time steps
            self.r[:,:,t] = np.sum(self.z[:,:,t-self.tau_ref:t], axis=-1)

            #integrate input and recurrent currents from spikes at previous time step
            i_in = np.matmul(self.in_weights, input[:,:,t-1])
            i_rec = np.matmul(self.rec_weights, self.z[:,:,t-1])

            #enforce the refractory period
            i_reset = -(self.v[:,:,t-1] + i_in + i_rec)*self.r[:,:,t]

            #update the voltage
            self.v[:,:,t] = (self.dt/self.tau)*(self.v[:,:,t-1]) + i_in + i_rec + i_reset

            #apply spike function to current time step
            self.z[:,:,t] = self.spike_function(self.v[:,:,t])

        #truncate zero padding for tau_ref
        state = (self.v[:,:,self.tau_ref:], self.z[:,:,self.tau_ref:], self.r[:,:,self.tau_ref:])
        
        return state


class InputConnectivityGenerator():

    def __init__(self,n_in,n_rec,p=0.1):

        self.p = p
        self.inputs, self.units = n_in, n_rec

    def run_generator(self):

        self.weights = np.zeros((self.units, self.inputs))
        self.k = int(round(self.p*self.inputs))
        for n in range(0, self.units):
            for a in range(0, self.k):
                rand = np.random.randint(0, self.inputs)
                while rand == n or self.weights[n][rand] == 1:
                    rand = np.random.randint(0, self.inputs)
                self.weights[n][rand] = 0.03

        return self.weights

class ExInConnectivityMatrixGenerator():

    def __init__(self, n_excite, n_inhib, p_xx, dtype=np.float32):

        # Determine numbers of neurons
        self.n_excite = n_excite
        self.n_inhib = n_inhib
        self.n_neurons = n_excite + n_inhib

        # Initialize weight matrix
        self.weights = np.zeros((self.n_neurons, self.n_neurons), dtype=dtype)

        self.p_ee = p_xx[0,0]; self.p_ie = p_xx[0,1]
        self.p_ei = p_xx[1,0]; self.p_ii = p_xx[1,1]

        # Calculate total number of connections per neuron
        self.k_ii = int(round(self.p_ii * (self.n_inhib - 1)))
        self.k_ei = int(round(self.p_ei * self.n_inhib))
        self.k_ie = int(round(self.p_ie * self.n_excite))
        self.k_ee = int(round(self.p_ee * (self.n_excite - 1)))

    def run_generator(self):

        """
        Each row can be considered as the incoming connections to a neuron
        i.e. each row is a postsynaptic cell and each element is a presynaptic
        cell
        """

        # E to E connections
        for n in range(0, self.n_excite):
            for a in range(0, self.k_ee):
                rand = np.random.randint(0, self.n_excite)
                while rand == n or self.weights[rand][n] == 1:
                    rand = np.random.randint(0, self.n_excite)
                self.weights[rand][n] = 0.03

        # E to I connections
        for n in range(0, self.n_excite):
            for a in range(0, self.k_ei):
                rand = np.random.randint(self.n_excite, self.n_excite + self.n_inhib)
                while self.weights[rand][n] == 1:
                    rand = np.random.randint(self.n_excite, self.n_excite + self.n_inhib)
                self.weights[rand][n] = 0.03

        # I to E connections
        for n in range(0, self.n_inhib):
            for a in range(0, self.k_ie):
                rand = np.random.randint(0, self.n_excite)
                while self.weights[rand][n + self.n_excite] == 1:
                    rand = np.random.randint(0, self.n_excite)
                self.weights[rand][n + self.n_excite] = -0.03

        # I to I connections
        for n in range(0, self.n_inhib):
            for a in range(0, self.k_ii):
                rand = np.random.randint(self.n_excite, self.n_excite + self.n_inhib)
                while rand == (n + self.n_excite) or self.weights[rand][n + self.n_excite] == 1:
                    rand = np.random.randint(self.n_excite, self.n_excite + self.n_inhib)
                self.weights[rand][n + self.n_excite] = -0.03

        return self.weights
