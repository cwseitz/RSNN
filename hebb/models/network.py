import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from numpy.random import default_rng
from networkx.generators.random_graphs import erdos_renyi_graph
from hebb.util import *


class GaussianNetwork:

    """
    This function generates a gaussian network from parameter maps
    over a 2D lattice. Generates a connectivity matrix Cij in O(n^2) time

    Parameters
    ----------
    N : int
        Total number of units in the network
    sigma : 2D ndarray
        Reach parameter for every neuron
    bias_ij : 2D ndarray
        Bias for connection direction i->j
    bias_ji : 2D ndarray
        Bias for connection direction j->i
    q : 2D ndarray
        Binomial probability of no synapse

    """

    def __init__(self, N, sigma, q, delta=1):

        #check sigma value to ensure reasonable connection probabilities
        # min_sig = 5
        # if sigma < min_sig:
        #      raise ValueError(f'Sigma value is below the minimum value {min_sig}')

        #check rho value
        # max_rho = sigma*np.sqrt(2*np.pi)*np.exp(delta**2/(2*sigma**2))
        # print(f'Maximum rho value: {max_rho}')
        # if rho > max_rho:
        #      raise ValueError(f'Rho value is beyond the maximum value {max_rho}')

        self.N = N
        self.M = int(round(np.sqrt(N)))
        self.delta = delta
        self.sigma = sigma
        self.q = q
        self.C = np.zeros((N,N))

        idx_x, idx_y = np.triu_indices(self.N, k=1) #upper triangle indices
        xv, yv = np.meshgrid(np.arange(self.M),np.arange(self.M))
        self.X, self.Y = xv.ravel(), yv.ravel()

        #iterate over upper triangle of connectivity matrix
        for k in range(idx_x.shape[0]):
            #get grid coordinates from conn matrix indices
            i = idx_x[k]; j = idx_y[k]
            r_i = self.X[i], self.Y[i] #neuron i grid coordinates
            r_j = self.X[j], self.Y[j] #neuron j grid coordinates
            dr_ij = torus_dist(r_i, r_j, self.M, delta=self.delta)
            k_ij = delta_gauss(dr_ij, self.sigma[r_i], self.delta)
            k_ji = delta_gauss(dr_ij, self.sigma[r_j], self.delta)
            syn = trinomial(k_ij, k_ij, self.q)
            if syn == 1:
                self.C[i,j] = 1
            elif syn == -1:
                self.C[j,i] = 1

class ExInGaussianNetwork:

    """
    This function generates a gaussian network from parameter maps
    over a 2D lattice. Generates a connectivity matrix Cij in O(n^2) time

    Parameters
    ----------
    N : int
        Total number of units in the network
    sigma : 2D ndarray
        Reach parameter for every neuron
    bias_ij : 2D ndarray
        Bias for connection direction i->j
    bias_ji : 2D ndarray
        Bias for connection direction j->i
    q : 2D ndarray
        Binomial probability of no synapse

    """

    def __init__(self, N, sigma_e, sigma_i, bias_e, bias_i, q, p_e=0.8, delta=1):

        #check bias values
        max_bias_e = np.exp(delta**2/(2*sigma_e**2))
        max_bias_i = np.exp(delta**2/(2*sigma_e**2))
        print(f'Maximum excitatory bias value: {max_bias_e}')
        print(f'Maximum inhibitory bias value: {max_bias_i}')
        if bias_e > max_bias_e:
             raise ValueError(f'Excitatory bias is beyond the maximum value {max_bias_e}')
        elif bias_i > max_bias_i:
             raise ValueError(f'Inhibitory bias is beyond the maximum value {max_bias_i}')

        self.N = N
        self.M = int(round(np.sqrt(N)))
        self.delta = delta
        self.sigma_e = sigma_e
        self.sigma_i = sigma_i
        self.bias_e = bias_e
        self.bias_i = bias_i
        self.q = q
        self.C = np.zeros((N,N))

        #Randomly select p_e*N neurons to be excitatory
        bb = np.arange(0,N,1)
        self.ex_idx = np.random.choice(bb, size=int(round(p_e*self.N)), replace=False)
        self.in_idx = np.setdiff1d(bb,self.ex_idx)

        #Get a meshgrid to convert between an (N,1) vector and a (M,M) grid
        xv, yv = np.meshgrid(np.arange(self.M),np.arange(self.M))
        self.X, self.Y = xv.ravel(), yv.ravel()

        #Get upper triangle indices
        idx_x, idx_y = np.triu_indices(self.N, k=1) #upper triangle indices

        #Build bias maps and sigma maps
        self.sigma = np.zeros((self.M,self.M))
        self.bias = np.zeros((self.M,self.M))
        self.sigma[self.X[self.ex_idx], self.Y[self.ex_idx]] = self.sigma_e
        self.sigma[self.X[self.in_idx], self.Y[self.in_idx]] = self.sigma_i
        self.bias[self.X[self.ex_idx], self.Y[self.ex_idx]] = self.bias_e
        self.bias[self.X[self.in_idx], self.Y[self.in_idx]] = self.bias_i

        #iterate over upper triangle of connectivity matrix
        for k in range(idx_x.shape[0]):
            #get grid coordinates from conn matrix indices
            i = idx_x[k]; j = idx_y[k]
            r_i = self.X[i], self.Y[i] #neuron i grid coordinates
            r_j = self.X[j], self.Y[j] #neuron j grid coordinates
            dr_ij = torus_dist(r_i, r_j, self.M, delta=self.delta)
            k_ij = self.bias[r_i]*delta_gauss(dr_ij, self.sigma[r_i], self.delta)
            k_ji = self.bias[r_j]*delta_gauss(dr_ij, self.sigma[r_j], self.delta)
            syn = trinomial(k_ij, k_ij, self.q)
            if syn == 1:
                self.C[i,j] = 1
            elif syn == -1:
                self.C[j,i] = 1


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
        self.C = np.zeros((self.n_neurons, self.n_neurons))

    def run_generator(self):
        self.generate_C()
        self.generate_XIJ()

    def generate_C(self):

        self.G = erdos_renyi_graph(self.n_neurons, p=self.p, directed=True)
        self.C = nx.convert_matrix.to_numpy_array(self.G)
        self.C[:self.n_excite,:self.n_excite] *= self.J_ee
        self.C[:self.n_excite,self.n_excite:] *= self.J_ei
        self.C[self.n_excite:,:self.n_excite] *= self.J_ie
        self.C[self.n_excite:,self.n_excite:] *= self.J_ii

    def generate_XIJ(self):

        self.k = int(round(self.n_neurons*self.p))
        self.XIJ = np.zeros((self.n_neurons, self.n_in))
        for n in range(0, self.n_in):
            rng = default_rng()
            idx = rng.choice(self.n_neurons, size=self.k, replace=False)
            self.XIJ[idx,n] = 1

    def get_colormat(self):
        self.color_mat = np.zeros_like(self.C)
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
        G = nx.convert_matrix.from_numpy_array(np.abs(self.C))
        idxs = np.argwhere(self.C != 0)

        self.get_colormat()
        for idx in idxs:
            x,y = idx
            color_idx = int(self.color_mat[x,y])
            G.edges[x,y]['color'] = colors[color_idx]

        colors = [G[u][v]['color'] for u,v in G.edges()]
        pos = nx.spring_layout(G)
        nx.draw(G, pos, ax=ax[0], alpha=0.1, node_size=5, node_color='black',
                edge_color=colors, with_labels=labels)
        ax[1].imshow(self.C, cmap='gray')
        colormap = cm.get_cmap('gray')
        map = mpl.cm.ScalarMappable(cmap=colormap)
        ax[1].imshow(self.C, cmap='gray')
        plt.colorbar(map, ax=ax[1], fraction=0.046, pad=0.04)

        ax[2].plot(in_bins[:-1], in_vals, color='red', label='In')
        ax[2].plot(out_bins[:-1], out_vals, color='blue', label='Out')
        ax[2].plot(in_out_bins[:-1], in_out_vals, color='cyan', label='In/Out')
        ax[2].legend()

        plt.tight_layout()
