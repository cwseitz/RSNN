import matplotlib.pyplot as plt
import numpy as np


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
