import numpy as np
import networkx as nx
from ..util import *

##################################################
## Library of network models for generating
## connectivity matrices for neural net simulations
##################################################
## Author: Clayton Seitz
## Copyright: 2021, The Hebb Project
## Email: cwseitz@uchicago.edu
##################################################

class GaussianNetwork:

    """
    This function generates a gaussian network from parameter maps
    over a 2D lattice. Generates a connectivity matrix Cij in O(n^2) time

    Neuron inputs are along axis=1 and neuron outputs are along axis=0
    in the connectivity (adjacency) matrix

    Parameters
    ----------
    N : int
        Total number of units in the network
    sigma : 2D ndarray
        Reach parameter for every neuron
    q : 2D ndarray
        Binomial probability of no synapse

    """

    def __init__(self, N, sigma, q, delta=1):

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
            dr_ij = tordist(r_i, r_j, self.M, delta=self.delta)
            k_ij = delta_gauss(dr_ij, self.sigma[r_i], self.delta)
            k_ji = delta_gauss(dr_ij, self.sigma[r_j], self.delta)
            syn = sample_trinomial(k_ij, k_ij, self.q)
            if syn == 1:
                self.C[i,j] = 1
            elif syn == -1:
                self.C[j,i] = 1

class ExInGaussianNetwork:

    """
    This function generates a gaussian network from parameter maps
    over a 2D lattice. Generates a connectivity matrix Cij in O(n^2) time

    Neuron inputs are along axis=1 and neuron outputs are along axis=0
    in the connectivity (adjacency) matrix


    Parameters
    ----------
    N : int
        Total number of units in the network
    sigma : 2D ndarray
        Reach parameter for every neuron
    q : 2D ndarray
        Binomial probability of no synapse

    """

    def __init__(self, N, sigma_e, sigma_i, q, p_e=0.8, delta=1):

        self.N = N
        self.M = int(round(np.sqrt(N)))
        self.delta = delta
        self.p_e = p_e
        self.sigma_e = sigma_e
        self.sigma_i = sigma_i
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

        #Build sigma maps
        self.sigma = np.zeros((self.M,self.M))
        self.sigma[self.X[self.ex_idx], self.Y[self.ex_idx]] = self.sigma_e
        self.sigma[self.X[self.in_idx], self.Y[self.in_idx]] = self.sigma_i

        #iterate over upper triangle of connectivity matrix
        for k in range(idx_x.shape[0]):
            #get grid coordinates from conn matrix indices
            i = idx_x[k]; j = idx_y[k]
            r_i = self.X[i], self.Y[i] #neuron i grid coordinates
            r_j = self.X[j], self.Y[j] #neuron j grid coordinates
            dr_ij = tordist(r_i, r_j, self.M, delta=self.delta)
            k_ij = delta_gauss(dr_ij, self.sigma[r_i], self.delta)
            k_ji = delta_gauss(dr_ij, self.sigma[r_j], self.delta)
            syn = sample_trinomial(k_ij, k_ji, self.q)
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
    Biosystems, Sporns 2006.

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

    Returns
    -------
    """

    def __init__(self, mx_lvl, E, sz_cl, seed=None):

        self.mx_lvl = mx_lvl
        self.E = E
        self.sz_cl = sz_cl
        self.seed = seed

        rng = self.get_rng(self.seed)
        t = np.ones((2, 2)) * 2
        n = 2**self.mx_lvl
        self.sz_cl -= 1

        for lvl in range(1, self.mx_lvl):
            s = 2**(lvl+1)
            self.C = np.ones((s, s))
            grp1 = range(int(s/2))
            grp2 = range(int(s/2), s)
            ix1 = np.add.outer(np.array(grp1) * s, grp1).flatten()
            ix2 = np.add.outer(np.array(grp2) * s, grp2).flatten()
            self.C.flat[ix1] = t
            self.C.flat[ix2] = t
            self.C += 1
            t = self.C.copy()

        self.C -= (np.ones((s, s)) + self.mx_lvl * np.eye(s))
        ee = self.mx_lvl - self.C - self.sz_cl
        ee = (ee > 0) * ee
        prob = (1 / self.E**ee) * (np.ones((s, s)) - np.eye(s))
        self.C = (prob > rng.random_sample((n, n)))
        k = np.sum(self.C)

        self.C = np.array(self.C, dtype=int)

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
