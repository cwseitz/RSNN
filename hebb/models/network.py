import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse import dok_matrix
from ..util import *

##################################################
## Library of network models for generating
## connectivity matrices for neural net simulations
##################################################
## Author: Clayton Seitz
## Copyright: 2021, The Hebb Project
## Email: cwseitz@uchicago.edu
##################################################

class ExInRandomNetwork:


    """
    This function generates a random network of excitatory and inhibitory
    neurons. N = N_e + N_e = p_e*N + p_i*N.

    Synaptic weights are scaled as 1/sqrt(N). This is not a binomial random
    graph. It is assumed that that probability of a synapse, for example,
    p_ee gives exactly C_ee = p_ee*N connections between and excitatory
    neuron and other excitatory neurons

    Parameters
    ----------
    """


    def __init__(self, N, p_e, n_in, p_xx, J_xx, q):

        # Determine numbers of neurons
        self.N = N
        self.p_e = p_e
        self.n_e = int(round(self.p_e*self.N))
        self.n_i = int(round((1-self.p_e)*self.N))
        self.n_in = n_in
        self.p_ee, self.p_ei, self.p_ie, self.p_ii = p_xx
        self.J_ee, self.J_ei, self.J_ie, self.J_ii = J_xx
        self.q = q

        # Initialize connectivity matrix
        self.C = np.zeros((self.N, self.N))
        self.k_ee = self.p_ee*np.ones((self.n_e,))
        self.k_ei = self.p_ei*np.ones((self.n_i,))
        self.k_ie = self.p_ie*np.ones((self.n_i,))

        #Excitatory-excitatory, excitatory-inhibitory
        for idx in range(self.n_e):
            s_ee = sample_trinomial(self.k_ee,self.k_ee,self.q*np.ones((self.n_e,)))
            s_ei = sample_trinomial(self.k_ei,self.k_ie,self.q*np.ones((self.n_i,)))
            self.C[:,idx] = np.concatenate((s_ee,s_ei))

        self.k_ii = self.p_ee*np.ones((self.n_i,))
        self.k_ei = self.p_ei*np.ones((self.n_e,))
        self.k_ie = self.p_ie*np.ones((self.n_e,))

        #Inhibitory-inhibitory, Inhibitory-excitatory
        for idx in range(self.n_e,self.N):
            s_ie = sample_trinomial(self.k_ie,self.k_ei,self.q*np.ones((self.n_e,)))
            s_ii = sample_trinomial(self.k_ii,self.k_ii,self.q*np.ones((self.n_i,)))
            self.C[:,idx] = np.concatenate((s_ie, s_ii))

        #Zero lower triangle and reflect negative values
        self.C[np.tril_indices(self.N,k=0)] = 0
        self.C = np.abs(np.clip(self.C,0,1) + np.clip(self.C.T,-1,0))

    def make_weighted(self):
        self.C[:self.n_e,:self.n_e] *= self.J_ee
        self.C[self.n_e:,:self.n_e] *= self.J_ei
        self.C[:self.n_e,self.n_e:] *= self.J_ie
        self.C[self.n_e:,self.n_e:] *= self.J_ii

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
    in the connectivity (adjacency) matrix'

    TODO: Figure out how to store connectivity matrix as a binary array
    (boolean arrays are 8-bit)


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
        self.ex_idx = np.random.choice(bb, size=int(round(p_e*N)),replace=False)
        self.in_idx = np.setdiff1d(bb,self.ex_idx)

        #Get a meshgrid to convert between an (N,1) vector and a (M,M) grid
        xv, yv = np.meshgrid(np.arange(self.M),np.arange(self.M), indexing='ij')
        X, Y = xv.ravel(), yv.ravel()
        self.C = np.zeros((N,N))
        q0 = np.ones_like((N,))

        #Excitatory-excitatory, excitatory-inhibitory
        for idx in self.ex_idx:

            x0,y0 = X[idx], Y[idx]
            k_ee = torgauss(X,Y,x0,y0,self.sigma_e)
            k_ee[X[self.in_idx],Y[self.in_idx]]= 0
            s_ee = sample_trinomial(k_ee,k_ee,self.q*q0)

            k_ei = torgauss(X,Y,x0,y0,self.sigma_e)
            k_ei[X[self.ex_idx],Y[self.ex_idx]]= 0
            k_ie = torgauss(X,Y,x0,y0,self.sigma_i)
            k_ie[X[self.ex_idx],Y[self.ex_idx]]= 0
            s_ei = sample_trinomial(k_ei,k_ie,self.q*q0)
            s = s_ee + s_ei
            self.C[:,idx] = s.flatten()

        #Inhibitory-inhibitory, Inhibitory-excitatory
        for idx in self.in_idx:

            x0,y0 = X[idx], Y[idx]
            k_ii = torgauss(X,Y,x0,y0,self.sigma_i)
            k_ii[X[self.ex_idx],Y[self.ex_idx]]= 0
            s_ii = sample_trinomial(k_ii,k_ii,self.q*q0)

            k_ie = torgauss(X,Y,x0,y0,self.sigma_i)
            k_ie[X[self.in_idx],Y[self.in_idx]]= 0
            k_ei = torgauss(X,Y,x0,y0,self.sigma_e)
            k_ei[X[self.in_idx],Y[self.in_idx]]= 0

            s_ie = sample_trinomial(k_ie,k_ei,self.q*q0)
            s = s_ii + s_ie
            self.C[:,idx] = s.flatten()


        #Zero lower triangle and reflect negative values
        self.C[np.tril_indices(N,k=0)] = 0
        self.C = np.abs(np.clip(self.C,0,1) + np.clip(self.C.T,-1,0))


    def make_weighted(self, J_ee, J_ei, J_ie, J_ii):

        ex_vec = np.zeros((self.N,)); ex_vec[self.ex_idx] = 1
        in_vec = np.zeros((self.N,)); in_vec[self.in_idx] = 1
        C_ee = np.einsum('i,ij->ij',in_vec,np.einsum('i,ji->ji',ex_vec,self.C))
        C_ei = np.einsum('i,ij->ij',ex_vec,np.einsum('i,ji->ji',ex_vec,self.C))
        C_ie = np.einsum('i,ij->ij',in_vec,np.einsum('i,ji->ji',in_vec,self.C))
        C_ii = np.einsum('i,ij->ij',ex_vec,np.einsum('i,ji->ji',in_vec,self.C))
        self.C = J_ee*C_ee + J_ei*C_ei + J_ie*C_ie + J_ii*C_ii

class ExInGaussianNetwork_Sparse:

    """
    This function generates a gaussian network from parameter maps
    over a 2D lattice. Generates a connectivity matrix Cij in O(n^2) time

    This function takes approximately twice as long to generate connectivity
    as ExInGaussianNetwork, but saves significant memory for large
    connectivity matrices. Use wisely!

    TODO: Figure out weighting scheme for sparse matrices


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
        self.ex_idx = np.random.choice(bb, size=int(round(p_e*N)),replace=False)
        self.in_idx = np.setdiff1d(bb,self.ex_idx)

        #Get a meshgrid to convert between an (N,1) vector and a (M,M) grid
        xv, yv = np.meshgrid(np.arange(self.M),np.arange(self.M), indexing='ij')
        X, Y = xv.ravel(), yv.ravel()
        self.C = dok_matrix((N,N),dtype=np.bool)
        q0 = np.ones_like((N,))
        self.adj_dict = {}

        #Excitatory-excitatory, excitatory-inhibitory
        for idx in self.ex_idx:

            x0,y0 = X[idx], Y[idx]
            k_ee = torgauss(X,Y,x0,y0,self.sigma_e)
            k_ee[X[self.in_idx],Y[self.in_idx]]= 0
            s_ee = sample_trinomial(k_ee,k_ee,self.q*q0)

            k_ei = torgauss(X,Y,x0,y0,self.sigma_e)
            k_ei[X[self.ex_idx],Y[self.ex_idx]]= 0
            k_ie = torgauss(X,Y,x0,y0,self.sigma_i)
            k_ie[X[self.ex_idx],Y[self.ex_idx]]= 0
            s_ei = sample_trinomial(k_ei,k_ie,self.q*q0)
            s = s_ee.flatten() + s_ei.flatten(); s[idx:] = 0

            tt = list(zip(np.where(s == 1)[0],list(idx*np.ones(s.size,np.int16))))
            self.adj_dict.update(dict.fromkeys(tt, 1))
            tt = list(zip(list(idx*np.ones(s.size,np.int16)),np.where(s == -1)[0]))
            self.adj_dict.update(dict.fromkeys(tt, 1))

        #Inhibitory-inhibitory, Inhibitory-excitatory
        for idx in self.in_idx:

            x0,y0 = X[idx], Y[idx]
            k_ii = torgauss(X,Y,x0,y0,self.sigma_i)
            k_ii[X[self.ex_idx],Y[self.ex_idx]]= 0
            s_ii = sample_trinomial(k_ii,k_ii,self.q*q0)

            k_ie = torgauss(X,Y,x0,y0,self.sigma_i)
            k_ie[X[self.in_idx],Y[self.in_idx]]= 0
            k_ei = torgauss(X,Y,x0,y0,self.sigma_e)
            k_ei[X[self.in_idx],Y[self.in_idx]]= 0

            s_ie = sample_trinomial(k_ie,k_ei,self.q*q0)
            s = s_ii.flatten() + s_ie.flatten(); s[idx:] = 0

            tt = list(zip(np.where(s == 1)[0],list(idx*np.ones(s.size,np.int16))))
            self.adj_dict.update(dict.fromkeys(tt, 1))
            tt = list(zip(list(idx*np.ones(s.size,np.int16)),np.where(s == -1)[0]))
            self.adj_dict.update(dict.fromkeys(tt, 1))

        self.C._update(self.adj_dict)

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
