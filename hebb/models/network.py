import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from numpy.random import default_rng
from networkx.generators.random_graphs import erdos_renyi_graph
from scipy.ndimage import gaussian_filter
from hebb.util import *

class SpatialNetwork2D:

    """
    This function generates a directed network where connection probabilities
    are a function of space. The lattice axial dimension should be M = np.sqrt(N)

    Parameters
    ----------

    Returns
    -------
    """

    def __init__(self, N, p, J_xx, sigma_e=1, sigma_i=2, delta=1, alpha=10):

        self.p = p
        self.N = N
        self.M = int(round(np.sqrt(N)))
        self.alpha = alpha
        self.delta = delta
        self.CIJ = np.zeros((self.M,self.M,N))
        self.J_ee, self.J_ei, self.J_ie, self.J_ii = J_xx
        self.in_idx = []
        self.make_grid()

        k = 0
        for i in range(self.M):
            for j in range(self.M):
                sigma = sigma_e
                if np.random.uniform(0,1) < p:
                    sigma = sigma_i
                    self.in_idx.append(k)
                f = np.zeros((self.M, self.M)); f[i,j] = self.alpha
                x = np.random.uniform(0,1,size=f.shape)
                g = gaussian_filter(f, sigma, mode='wrap')
                self.CIJ[:,:,k] = np.array(x < g, dtype=np.int32)
                k+=1

        self.CIJ = np.reshape(self.CIJ, (self.N, self.N))
        idx_x, idx_y = np.triu_indices(self.N) #upper triangle indices
        for k in range(len(idx_x)):
            i,j = idx_x[k], idx_y[k]
            #check for a bidirectional synapse
            if self.CIJ[i,j] == 1 and self.CIJ[j,i] == 1:
                if np.random.binomial(1, 0.5) == 1:
                    self.CIJ[j,i] = 0 #make neuron j the presynaptic neuron
                    self.CIJ[i,j] *= self.get_psp(i,j,j)
                else:
                    self.CIJ[i,j] = 0 #make neuron j the postsynaptic neuron
                    self.CIJ[j,i] *= self.get_psp(i,j,i)

    def get_psp(self, i, j, pre):

        """
        Get the PSP based on neuron indices and the
        known indices of inhibitory neurons
        """

        if i not in self.in_idx and j not in self.in_idx:
            return self.J_ee
        elif i in self.in_idx and j in self.in_idx:
            return self.J_ii
        elif i in self.in_idx and j not in self.in_idx and pre == i:
            return self.J_ie
        elif i in self.in_idx and j not in self.in_idx and pre == j:
            return self.J_ei
        elif i not in self.in_idx and j in self.in_idx and pre == i:
            return self.J_ei
        elif i not in self.in_idx and j in self.in_idx and pre == j:
            return self.J_ie


    def make_grid(self):

        X = np.arange(0, self.M, self.delta)
        Y = np.arange(0, self.M, self.delta)
        self.X, self.Y = np.meshgrid(X, Y)
        self.r = np.empty(self.X.shape + (2,))
        self.r[:,:,0] = self.X
        self.r[:,:,1] = self.Y

    def pairwise_stats(self, min_sep, max_sep, min_sig, max_sig, rho_1=0.1, rho_2=0.1):

        """
        Statistics of pairwise connectivity
        (min_sep, max_sep) are integer multiples of the grid spacing delta
        """

        def f(mu_1, sigma_1, mu_2, sigma_2, rho_1, rho_2):

            f_1 = rho_1*multi_gauss(self.r, mu_1, sigma_1)
            f_2 = rho_2*multi_gauss(self.r, mu_2, sigma_2)
            f_12 = f_1*f_2
            return f_1, f_2, f_12

        def g(sep, sig):
            mu = np.array([sep,0])
            sigma = np.array([[sig,0],[0,sig]])
            return mu, sigma

        seps = self.delta*np.arange(min_sep, max_sep, 5) #range for the distance between neurons
        sigs = self.delta*np.arange(min_sig, max_sig, 5) #range for the broadness of connections
        self.N = np.zeros((len(sigs), len(seps)))
        self.N_var = np.zeros_like(self.N)

        for i, sig in enumerate(sigs):
            for j, sep in enumerate(seps):
                mu_1, sigma_1 = g(sep, sig)
                mu_2, sigma_2 = g(-sep, sig)
                f_1, f_2, f_12 = f(mu_1, sigma_1, mu_2, sigma_2, rho_1, rho_2)
                self.N[i,j] = np.sum(f_1+f_2)
                self.N_var[i,j] = np.sum(f_12*(1-f_12))

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
    (sometimes called a Brunel network) and an input connectivity matrix.

    The network topology is a fixed in-degree network where the in-degree
    is specified by the user. The out-degree varies

    Parameters
    ----------

    Returns
    -------

    """

    def __init__(self, n_excite, n_inhib, n_in, p, J_xx):

        # Determine numbers of neurons
        self.n_excite = n_excite
        self.n_inhib = n_inhib
        self.n_in = n_in
        self.n_neurons = n_excite + n_inhib
        self.p = p
        self.J_ee, self.J_ei, self.J_ie, self.J_ii = J_xx

        # Initialize connectivity matrix
        self.CIJ = np.zeros((self.n_neurons, self.n_neurons))

    def run_generator(self):
        self.generate_CIJ()
        self.generate_XIJ()

    def generate_CIJ(self):

        self.G = erdos_renyi_graph(self.n_neurons, p=self.p, directed=True)
        self.CIJ = nx.convert_matrix.to_numpy_array(self.G)
        self.CIJ[:self.n_excite,:self.n_excite] *= self.J_ee
        self.CIJ[:self.n_excite,self.n_excite:] *= self.J_ei
        self.CIJ[self.n_excite:,:self.n_excite] *= self.J_ie
        self.CIJ[self.n_excite:,self.n_excite:] *= self.J_ii

    def generate_XIJ(self):

        self.k = int(round(self.n_neurons*self.p))
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

        in_deg = [self.G.in_degree(n) for n in self.G.nodes()]
        out_deg = [self.G.out_degree(n) for n in self.G.nodes()]
        in_out = [in_deg[i]/out_deg[i] for i in range(len(in_deg))]
        in_vals, in_bins = np.histogram(in_deg)
        out_vals, out_bins = np.histogram(out_deg)
        in_out_vals, in_out_bins = np.histogram(in_out)

        fig, ax = plt.subplots(1,3)
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

        ax[2].plot(in_bins[:-1], in_vals, color='red', label='In')
        ax[2].plot(out_bins[:-1], out_vals, color='blue', label='Out')
        ax[2].plot(in_out_bins[:-1], in_out_vals, color='cyan', label='In/Out')
        ax[2].legend()

        plt.tight_layout()
