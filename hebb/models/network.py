import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from numpy.random import default_rng

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

    def run_generator(self, scale=False):

        rng = self.get_rng(self.seed)
        t = np.ones((2, 2)) * 2
        n = 2**self.mx_lvl
        self.sz_cl -= 1

        for lvl in range(1, self.mx_lvl):
            s = 2**(lvl+1)
            self.J = np.ones((s, s))
            grp1 = range(int(s/2))
            grp2 = range(int(s/2), s)
            ix1 = np.add.outer(np.array(grp1) * s, grp1).flatten()
            ix2 = np.add.outer(np.array(grp2) * s, grp2).flatten()
            self.J.flat[ix1] = t
            self.J.flat[ix2] = t
            self.J += 1
            t = self.J.copy()

        self.J -= (np.ones((s, s)) + self.mx_lvl * np.eye(s))
        ee = self.mx_lvl - self.J - self.sz_cl
        ee = (ee > 0) * ee
        prob = (1 / self.E**ee) * (np.ones((s, s)) - np.eye(s))
        self.J = (prob > rng.random_sample((n, n)))
        k = np.sum(self.J)

        self.J = np.array(self.J, dtype=int)

        if scale:
            self.J = self.J/n

        return self.J

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

        fix, ax = plt.subplots(1,2)
        colormap = cm.get_cmap('gray')
        map = mpl.cm.ScalarMappable(cmap=colormap)
        ax[0].imshow(self.J, cmap='gray')
        plt.colorbar(ax=ax[0])

        G = nx.convert_matrix.from_numpy_array(self.J)
        idxs = np.argwhere(self.J > 0)
        self.level_mat = self.level_mat()

        if self.color_by == 'level':
            if colors is None:
                colors = cm.coolwarm(np.linspace(0,1,self.mx_lvl-self.sz_cl))
            for idx in idxs:
                x,y = idx
                color_idx = int(self.level_mat[x,y])
                G.edges[x,y]['color'] = colors[color_idx]

        colors = [G[u][v]['color'] for u,v in G.edges()]
        pos = nx.spring_layout(G)
        nx.draw(G, pos, ax=ax[1], alpha=0.1, node_size=5, node_color='black',
                edge_color=colors, with_labels=labels)
        plt.tight_layout()

class BrunelNetwork:

    """
    This function generates a directed network of excitatory and inhibitory
    (sometimes called a Brunel network) and an input connectivity matrix

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

    def __init__(self, n_excite, n_inhib, n_in, p, p_ee, p_ei, p_ie, p_ii):

        # Determine numbers of neurons
        self.n_excite = n_excite
        self.n_inhib = n_inhib
        self.n_in = n_in
        self.n_neurons = n_excite + n_inhib

        # Initialize connectivity matrix
        self.CIJ = np.zeros((self.n_neurons, self.n_neurons))

        # Calculate total number of connections per neuron
        self.k = int(round(p*self.n_neurons))
        self.k_ii = int(round(p_ii * (self.n_inhib - 1)))
        self.k_ei = int(round(p_ei * self.n_inhib))
        self.k_ie = int(round(p_ie * self.n_excite))
        self.k_ee = int(round(p_ee * (self.n_excite - 1)))

    def run_generator(self):
        self.generate_CIJ()
        self.generate_XIJ()

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
                self.CIJ[n + self.n_excite][rand] = -1

        # I to I connections
        for n in range(0, self.n_inhib):
            for a in range(0, self.k_ii):
                rand = np.random.randint(self.n_excite, self.n_excite + self.n_inhib)
                while rand == (n + self.n_excite) or self.CIJ[n + self.n_excite][rand] == 1:
                    rand = np.random.randint(self.n_excite, self.n_excite + self.n_inhib)
                self.CIJ[n + self.n_excite][rand] = -1

    def generate_XIJ(self):

        self.XIJ = np.zeros((self.n_neurons, self.n_in))
        for n in range(0, self.n_in):
            rng = default_rng()
            idx = rng.choice(self.n_neurons, size=self.k, replace=False)
            self.XIJ[idx,n] = 1

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
        G = nx.convert_matrix.from_numpy_array(np.abs(self.CIJ))
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
        ax[1].imshow(self.CIJ, cmap='gray')
        colormap = cm.get_cmap('gray')
        map = mpl.cm.ScalarMappable(cmap=colormap)
        ax[1].imshow(self.CIJ, cmap='gray')
        plt.colorbar(map, ax=ax[1], fraction=0.046, pad=0.04)

        plt.tight_layout()
