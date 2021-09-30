import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm

class InputConnectivityGenerator:

    """
    Generate the connectivity matrix for external stimulation
    Defaults to a diagonal matrix

    Parameters
    ----------
    n_rec : int,
    """

    def __init__(self, n_rec, mu, sigma):

        self.n_rec = n_rec
        self.mu = mu
        self.sigma = sigma
        self.X = self.generate_conn_mat()

    def generate_conn_mat(self):

        for n in range(0, self.n_neurons):
            for a in range(0, self.k):
                rand = numpy.random.randint(0, self.n_neurons)
                while rand == n or self.conn_mat[n][rand] == 1:
                    rand = numpy.random.randint(0, self.n_neurons)
                self.conn_mat[n][rand] = 1


    def make_weighted(self):

        # Generate random weights and fill matrix
        for i in range(0, self.n_neurons):
            for j in range(0, self.n_neurons):
                if self.X[i][j] == 1:
                    self.weight_mat[i][j] = (
                        numpy.random.lognormal(self.mu, self.sigma)
                    )

    def plot(self):
        fig, ax = plt.subplots()
        if self.X is None:
            raise ValueError('Generator function has not been called')
        ax.imshow(self.X, cmap='gray')
        plt.show()

class FractalNetwork:

    """
    This function generates a directed network with a hierarchical modular
    organization. All modules are fully connected and connection density
    decays as 1/(E^n), with n = index of hierarchical level.

    Generate a self-similiar connectivity matrix as outlined in
    Biosystems, Sporns 2006. This code was derived from the Brain Connectivity
    Toolbox in Python (a port of the original MATLAB code by Sporns et al. to Python)

    Parameters
    ----------
    mx_lvl : int
        number of hierarchical levels, N = 2^mx_lvl
    E : int
        connection density fall off per level
    sz_cl : int
        size of clusters (must be power of 2)
    seed : hashable, optional
        If None (default), use the np.random's global random state to generate random numbers.
        Otherwise, use a new np.random.RandomState instance seeded with the given value.
    color_by : str, optional
        Whether to color by level or largest group

    Returns
    -------
    J0 : NxN np.ndarray
        connection matrix
    K : int
        number of connections present in output J0
    """

    def __init__(self, mx_lvl, E, sz_cl, color_by='level', seed=None):

        self.mx_lvl = mx_lvl
        self.E = E
        self.sz_cl = sz_cl
        self.seed = seed
        self.color_by = color_by

    def get_rng(self, seed=None):

        if seed is None or seed == np.random:
            return np.random.mtrand._rand
        elif isinstance(seed, np.random.RandomState):
            return seed
        try:
            rstate =  np.random.RandomState(seed)
        except ValueError:
            rstate = np.random.RandomState(random.Random(seed).randint(0, 2**32-1))
        return rstate

    def run_generator(self):

        rng = self.get_rng(self.seed)
        t = np.ones((2, 2)) * 2
        n = 2**self.mx_lvl
        self.sz_cl -= 1

        for lvl in range(1, self.mx_lvl):
            s = 2**(lvl+1)
            self.J0 = np.ones((s, s))
            grp1 = range(int(s/2))
            grp2 = range(int(s/2), s)
            ix1 = np.add.outer(np.array(grp1) * s, grp1).flatten()
            ix2 = np.add.outer(np.array(grp2) * s, grp2).flatten()
            self.J0.flat[ix1] = t
            self.J0.flat[ix2] = t
            self.J0 += 1
            t = self.J0.copy()

        self.J0 -= (np.ones((s, s)) + self.mx_lvl * np.eye(s))
        ee = self.mx_lvl - self.J0 - self.sz_cl
        ee = (ee > 0) * ee
        prob = (1 / self.E**ee) * (np.ones((s, s)) - np.eye(s))
        self.J0 = (prob > rng.random_sample((n, n)))
        k = np.sum(self.J0)

        return np.array(self.J0, dtype=int)

    def level_mat(self):
        level_mat = np.zeros((2**self.mx_lvl,2**self.mx_lvl), dtype=np.int8)
        i = 0
        for k in range(self.sz_cl+1,self.mx_lvl+1):
            i += 1
            for n in range(2**(self.mx_lvl-k)):
                level_mat[n*2**k:, :n*2**k] = i
                level_mat[:n*2**k, n*2**k:] = i
        return level_mat


    def plot(self, colors=None, labels=False):

        """
        Visualize the connectivity graph. Edges can be colored in two modes:
        (1) by level and (2) by largest group

        Parameters
        ----------
        labels : bool,
            Whether or not to show node labels, Defaults to False
        """

        G = nx.convert_matrix.from_numpy_array(self.J0)
        idxs = np.argwhere(self.J0 > 0)
        self.level_mat = self.level_mat()

        if self.color_by == 'level':
            if colors is None:
                colors = cm.viridis(np.linspace(0,1,self.mx_lvl-self.sz_cl))
            for idx in idxs:
                x,y = idx
                color_idx = int(self.level_mat[x,y])
                G.edges[x,y]['color'] = colors[color_idx]

        colors = [G[u][v]['color'] for u,v in G.edges()]
        pos = nx.spring_layout(G)
        fig, ax = plt.subplots()
        nx.draw(G, pos, ax=ax, alpha=0.1, node_size=5, node_color='black',
                edge_color=colors, with_labels=labels)
        plt.tight_layout()

class BrunelNetwork:

    """
    This function generates a directed network of excitatory and inhibitory
    (sometimes called a Brunel network)

    Parameters
    ----------
    mx_lvl : int
        number of hierarchical levels, N = 2^mx_lvl
    E : int
        connection density fall off per level
    sz_cl : int
        size of clusters (must be power of 2)
    seed : hashable, optional
        If None (default), use the np.random's global random state to generate random numbers.
        Otherwise, use a new np.random.RandomState instance seeded with the given value.
    color_by : str, optional
        Whether to color by level or largest group

    Returns
    -------
    J0 : NxN np.ndarray
        connection matrix
    K : int
        number of connections present in output J0
    """

    def __init__(self, n_excite, n_inhib, p_ee, p_ei, p_ie, p_ii, mu, sigma):

        # Determine numbers of neurons
        self.n_excite = n_excite
        self.n_inhib = n_inhib
        self.n_neurons = n_excite + n_inhib
        self.mu = mu
        self.sigma = sigma

        # Initialize connectivity matrix
        self.CIJ = np.zeros((self.n_neurons, self.n_neurons))

        # Calculate total number of connections per neuron (remove
        # neuron from target if included (ee and ii))
        self.k_ii = int(round(p_ii * (self.n_inhib - 1)))
        self.k_ei = int(round(p_ei * self.n_inhib))
        self.k_ie = int(round(p_ie * self.n_excite))
        self.k_ee = int(round(p_ee * (self.n_excite - 1)))

    def run_generator(self):
        self.generate_CIJ()

    def generate_CIJ(self):

        # E to E connections
        for n in range(0, self.n_excite):
            for a in range(0, self.k_ee):
                rand = np.random.randint(0, self.n_excite)
                while rand == n or self.CIJ[n][rand] == 1:
                    rand = np.random.randint(0, self.n_excite)
                self.CIJ[n][rand] = 1

        # E to I connections
        for n in range(0, self.n_excite):
            for a in range(0, self.k_ei):
                rand = np.random.randint(self.n_excite, self.n_excite + self.n_inhib)
                while self.CIJ[n][rand] == 1:
                    rand = np.random.randint(self.n_excite, self.n_excite + self.n_inhib)
                self.CIJ[n][rand] = 1

        # I to E connections
        for n in range(0, self.n_inhib):
            for a in range(0, self.k_ie):
                rand = np.random.randint(0, self.n_excite)
                while self.CIJ[n + self.n_excite][rand] == 1:
                    rand = np.random.randint(0, self.n_excite)
                self.CIJ[n + self.n_excite][rand] = 1

        # I to I connections
        for n in range(0, self.n_inhib):
            for a in range(0, self.k_ii):
                rand = np.random.randint(self.n_excite, self.n_excite + self.n_inhib)
                while rand == (n + self.n_excite) or self.CIJ[n + self.n_excite][rand] == 1:
                    rand = np.random.randint(self.n_excite, self.n_excite + self.n_inhib)
                self.CIJ[n + self.n_excite][rand] = 1

    def make_weighted(self):


        # Generate random weights and fill matrix
        for i in range(0, self.n_neurons):
            for j in range(0, self.n_neurons):
                if self.CIJ[i][j] == 1:
                    self.CIJ[i][j] = (np.random.lognormal(self.mu, self.sigma))
                    # Make all I 10 times stronger AND NEGATIVE
                    if self.n_neurons > i > (self.n_neurons - self.n_inhib):
                        self.CIJ[i][j] *= -10



    def get_colormat(self):
        self.color_mat = np.zeros_like(self.CIJ)
        self.color_mat[self.n_excite:, :] = 1

    def plot(self, labels=False, colors=['red', 'blue']):

        """
        Visualize the connectivity graph

        Parameters
        ----------
        """

        fig, ax = plt.subplots(1,2)
        G = nx.convert_matrix.from_numpy_array(self.CIJ)
        idxs = np.argwhere(self.CIJ != 0)

        self.get_colormat()
        for idx in idxs:
            x,y = idx
            color_idx = int(self.color_mat[x,y])
            G.edges[x,y]['color'] = colors[color_idx]

        colors = [G[u][v]['color'] for u,v in G.edges()]
        pos = nx.spring_layout(G)
        nx.draw(G, pos, ax=ax[0], alpha=0.1, node_size=5, node_color='black',
                edge_color=colors, with_labels=labels)
        ax[1].imshow(self.CIJ)
        plt.tight_layout()
