import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm
from networkx.generators.random_graphs import watts_strogatz_graph

class InputConnectivityGenerator:

    """
    Generate the connectivity matrix for external stimulation
    Defaults to a diagonal matrix

    Parameters
    ----------
    n_rec : int,
    """

    def __init__(self, n_rec):
        self.n_rec = n_rec
        self.X = None

    def run_generator(self):
        self.X = np.eye(self.n_rec)
        #self.X = np.ones((self.n_rec, self.n_rec))
        return self.X

    def plot(self):
        fig, ax = plt.subplots()
        if self.X is None:
            raise ValueError('Generator function has not been called')
        ax.imshow(self.X, cmap='gray')
        plt.show()

class FractalConnect:

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

        return np.array(self.J0, dtype=int), k

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
