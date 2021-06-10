import matplotlib.pyplot as plt
import numpy as np


class InputConnectivityGenerator():

    def __init__(self,n_in,n_rec,p=0.1):

        self.p = p
        self.inputs, self.units = n_in, n_rec

    def run_generator(self):

        self.conn = self.conn()
        self.draw = self.draw_weights()
        self.weights = np.multiply(self.conn, self.draw)

        return self.weights

    def conn(self, dtype=np.float32):
        _conn = np.zeros((self.inputs, self.units))
        for i in range(self.inputs):
            _conn[i] = np.random.choice([0,1], p=[1-self.p, self.p], size=(self.units,))
        _conn = _conn.astype(dtype)
        return _conn

    def draw_weights(self, mu=-0.64, sigma=0.5, dtype=np.float32):

        weights = np.zeros_like(self.conn)
        nonzero = np.argwhere(self.conn > 0)
        nonzero_count = len(nonzero)

        draw = np.random.lognormal(mu, sigma, size=(nonzero_count,)).astype(dtype)

        for n in range(nonzero_count):
            x = nonzero[n][0]; y = nonzero[n][1]
            weights[x,y] = draw[n]

        return weights



class ExInConnectivityMatrixGenerator(object):

    def __init__(self, n_excite, n_inhib, p_ee, p_ei, p_ie, p_ii, mu, sigma, dtype=np.float32):

        # Determine numbers of neurons
        self.n_excite = n_excite
        self.n_inhib = n_inhib
        self.n_neurons = n_excite + n_inhib

        self.mu = mu
        self.sigma = sigma

        # Initialize connectivity matrix
        self.conn = np.zeros((self.n_neurons, self.n_neurons), dtype=dtype)

        # Initialize weight matrix
        self.weights = np.zeros((self.n_neurons, self.n_neurons), dtype=dtype)

        # Calculate total number of connections per neuron (remove
        # neuron from target if included (ee and ii))
        self.k_ii = int(round(p_ii * (self.n_inhib - 1)))
        self.k_ei = int(round(p_ei * self.n_inhib))
        self.k_ie = int(round(p_ie * self.n_excite))
        self.k_ee = int(round(p_ee * (self.n_excite - 1)))

    def run_generator(self):

        try:

            # Generate connectivity matrix and check it's successful
            if not self.generate_conn_mat():
                raise Exception('failed to generate connectivity matrix')
            # logging.info("generated E/I connectivity matrix")

            # Generate weight matrix and check that it's successful
            if not self.make_weighted():
                raise Exception('failed to weight connectivity matrix')
            # logging.info("weighted connectivity matrix")

            return self.weights/10 # again doing the hard-coded divide by 10 to make weights in the range that seems most trainable

        except Exception as e:
            # logging.exception(e)
            return False

    def generate_conn_mat(self):

        try:

            # E to E connections
            for n in range(0, self.n_excite):
                for a in range(0, self.k_ee):
                    rand = np.random.randint(0, self.n_excite)
                    while rand == n or self.conn[n][rand] == 1:
                        rand = np.random.randint(0, self.n_excite)
                    self.conn[n][rand] = 1

            # E to I connections
            for n in range(0, self.n_excite):
                for a in range(0, self.k_ei):
                    rand = np.random.randint(self.n_excite, self.n_excite + self.n_inhib)
                    while self.conn[n][rand] == 1:
                        rand = np.random.randint(self.n_excite, self.n_excite + self.n_inhib)
                    self.conn[n][rand] = 1

            # I to E connections
            for n in range(0, self.n_inhib):
                for a in range(0, self.k_ie):
                    rand = np.random.randint(0, self.n_excite)
                    while self.conn[n + self.n_excite][rand] == 1:
                        rand = np.random.randint(0, self.n_excite)
                    self.conn[n + self.n_excite][rand] = 1

            # I to I connections
            for n in range(0, self.n_inhib):
                for a in range(0, self.k_ii):
                    rand = np.random.randint(self.n_excite, self.n_excite + self.n_inhib)
                    while rand == (n + self.n_excite) or self.conn[n + self.n_excite][rand] == 1:
                        rand = np.random.randint(self.n_excite, self.n_excite + self.n_inhib)
                    self.conn[n + self.n_excite][rand] = 1

            return True

        except Exception as e:
            # logging.exception(e)
            return False

    def make_weighted(self):

        try:

            # Generate random weights and fill matrix
            for i in range(0, self.n_neurons):
                for j in range(0, self.n_neurons):
                    if self.conn[i][j] == 1:
                        self.weights[i][j] = (np.random.lognormal(self.mu, self.sigma))
                        # Make all I 10 times stronger AND NEGATIVE
                        if self.n_neurons > i > (self.n_neurons - self.n_inhib):
                            self.weights[i][j] = - self.weights[i][j] * 10

            return True

        except Exception as e:
            # logging.exception(e)
            return False


class ConnectivityMatrixGenerator(object):

    def __init__(self, n_neurons, p, mu, sigma, dtype=np.float32):

        self.n_neurons = n_neurons

        # Initialize connectivity matrix
        self.conn = np.zeros((self.n_neurons, self.n_neurons), dtype=dtype)

        # Initialize weight matrix
        self.weights = np.zeros((self.n_neurons, self.n_neurons), dtype=dtype)

        self.mu = mu
        self.sigma = sigma

        # Calculate total number of connections per neuron (remove
        # neuron from target if included (ee and ii))
        self.k = int(round(p * (self.n_neurons - 1)))

    def run_generator(self):
        try:
            # Generate connectivity matrix and check it's successful
            if not self.generate_conn_mat():
                raise Exception('failed to generate connectivity matrix')
            logging.info("connectivity matrix generated")

            # Generate weight matrix and check that it's successful
            if not self.make_weighted():
                raise Exception('failed to weight connectivity matrix')
            logging.info("connectivity matrix weighted")

            return self.weights/10 # hard-coded decrease weights by /10;
            # later on we will need to change the mu and sigma to
            # reflect current rather than conductance anyway

        except Exception as e:
            # logging.exception(e)
            return False

    def generate_conn_mat(self):

        try:
            for n in range(0, self.n_neurons):
                for a in range(0, self.k):
                    rand = np.random.randint(0, self.n_neurons)
                    while rand == n or self.conn[n][rand] == 1:
                        rand = np.random.randint(0, self.n_neurons)
                    self.conn[n][rand] = 1

            return True

        except Exception as e:
            # logging.exception(e)
            return False

    def make_weighted(self):

        try:

            # Generate random weights and fill matrix
            for i in range(0, self.n_neurons):
                for j in range(0, self.n_neurons):
                    if self.conn[i][j] == 1:
                        self.weights[i][j] = (
                            np.random.lognormal(self.mu, self.sigma)
                        )

            return True

        except Exception as e:
            # logging.exception(e)
            return False
