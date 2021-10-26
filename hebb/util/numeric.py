import numpy as np
from ..models import *
from .math import *

##################################################
## Library of numerical solutions to problems
## associated with Gaussian networks
##################################################
## Author: Clayton Seitz
## Copyright: 2021, The Hebb Project
## Email: cwseitz@uchicago.edu
##################################################

"""
Homogeneous Gaussian network
"""

def gauss_net_avg_deg(N, sigma, q, delta):

    """
    Average degree (in or out) of a homogeneous gaussian network for a single
    (sigma, q) parameter pair

    Parameters
    ----------
    N : int
        Number of neurons in the network. Must be a perfect square
    sigma : ndarray
        Standard deviation of the gaussian connectivity kernel
    q : ndarray
        Sparsity parameter
    delta: int,
        Lattice spacing

    """

    M = int(round(np.sqrt(N)))
    xv, yv = np.meshgrid(np.arange(M),np.arange(M))
    X, Y = xv.ravel(), yv.ravel()
    dvec = np.array([tordist((0,0),(X[i],Y[i]),M,delta) for i in range(N)])[1:]
    p_vec = np.zeros_like(dvec)
    for i, dr_ij in enumerate(dvec):
        k_ij = delta_gauss(dr_ij, sigma, delta)
        p_ij, p_ji, p_x  = trinomial(k_ij, k_ij, q)
        p_vec[i] = p_ij
    return np.sum(p_vec)

def gauss_net_var_deg(N, sigma, q, delta):

    """
    Variance in  degree (in or out) of a homogeneous gaussian network for a single
    (sigma, q) parameter pair

    Parameters
    ----------
    N : int
        Number of neurons in the network. Must be a perfect square
    sigma : ndarray
        Standard deviation of the gaussian connectivity kernel
    q : ndarray
        Sparsity parameter
    delta: int,
        Lattice spacing

    """

    M = int(round(np.sqrt(N)))
    xv, yv = np.meshgrid(np.arange(M),np.arange(M))
    X, Y = xv.ravel(), yv.ravel()
    dvec = np.array([tordist((0,0),(X[i],Y[i]),M,delta) for i in range(N)])[1:]
    p_vec = np.zeros_like(dvec)
    for i, dr_ij in enumerate(dvec):
        k_ij = delta_gauss(dr_ij, sigma, delta)
        p_ij, p_ji, p_x  = trinomial(k_ij, k_ij, q)
        p_vec[i] = p_ij*(1-p_ij)

    return np.sum(p_vec)

def gauss_net_deg_fixsig(N, sigma, qs, delta=1):

    """
    First two moments of the degree distribution (in or out) for fixed
    sigma, varying the sparsity parameter

    Parameters
    ----------
    N : int
        Number of neurons in the network. Must be a perfect square
    sigma : ndarray
        Standard deviation of the gaussian connectivity kernel
    qs : ndarray
        Values of the sparsity parameter to use
    delta: int,
        Lattice spacing

    """

    avg_arr = np.zeros((len(qs),))
    var_arr = np.zeros((len(qs),))
    for i,q in enumerate(qs):
        avg_arr[i] = gauss_net_avg_deg(N, sigma, q, delta)
        var_arr[i] = gauss_net_var_deg(N, sigma, q, delta)
    return avg_arr, var_arr

def gauss_net_deg_fixq(N, sigmas, q, delta=1):

    """
    First two moments of the degree distribution (in or out) for fixed
    sparsity, varying the sigma (reach) parameter

    Parameters
    ----------
    N : int
        Number of neurons in the network. Must be a perfect square
    sigmas : ndarray
        Standard deviations of the gaussian connectivity kernel
    q : ndarray
        Value of the sparsity parameter
    delta: int,
        Lattice spacing

    """

    avg_arr = np.zeros((len(sigmas),))
    var_arr = np.zeros((len(sigmas),))
    for i,sigma in enumerate(sigmas):
        avg_arr[i] = gauss_net_avg_deg(N, sigma, q, delta)
        var_arr[i] = gauss_net_var_deg(N, sigma, q, delta)
    return avg_arr, var_arr

def gauss_net_deg_full(N, sigmas, qs, delta=1):

    """
    First two moments of the degree distribution (in or out) over the entire
    parameter space (sigma, q)

    Parameters
    ----------
    N : int
        Number of neurons in the network. Must be a perfect square
    sigmas : ndarray
        Standard deviations of the gaussian connectivity kernel
    qs : ndarray
        Values of the sparsity parameter to use
    delta: int,
        Lattice spacing

    """

    avg_arr = np.zeros((len(qs),len(sigmas)))
    var_arr = np.zeros((len(qs),len(sigmas)))
    for i,q in enumerate(qs):
        for j, sigma in enumerate(sigmas):
            avg_arr[i,j] = gauss_net_avg_deg(N, sigma, q, delta)
            var_arr[i,j] = gauss_net_var_deg(N, sigma, q, delta)
    return avg_arr, var_arr

def gauss_net_shared(N, dvec, sigma, q, delta=1):

    """
    Find the average number of shared outputs (or inputs) as a function of distance
    between two neurons in a homogeneous gaussian network

    Parameters
    ----------
    N : int
        Number of neurons in the network. Must be a perfect square
    dvec : ndarray
        Distances over which to calculate average shared ouputs (inputs)
    sigma : ndarray
        Standard deviations of the gaussian connectivity kernel
    q : ndarray
        Value of the sparsity parameter
    delta: int,
        Lattice spacing

    """

    M = int(round(np.sqrt(N)))
    xv, yv = np.meshgrid(np.arange(M),np.arange(M))
    X, Y = xv.ravel(), yv.ravel()
    p_vec = np.zeros_like(dvec)
    for i, dr_ij in enumerate(dvec):
        k_ij_1 = torgauss(N, 0, 0, sigma, delta)
        #Move the second neuron along the diagonal to minimize 'bleed over' from pbc
        k_ij_2 = torgauss(N, dr_ij/np.sqrt(2), dr_ij/np.sqrt(2), sigma, delta)
        p_ij_1, p_ji_1, p_x_1 = trinomial(k_ij_1, k_ij_1, q)
        p_ij_2, p_ji_2, p_x_2 = trinomial(k_ij_2, k_ij_2, q)
        p_vec[i] = np.sum(p_ij_1*p_ij_2)

    return p_vec

def gauss_net_shared_exp(net, delta=1):

    """
    Find the average number of shared outputs (or inputs) as a function of distance
    between two neurons in a homogeneous gaussian network using connectivity
    matrices directly (experimental solution)

    Parameters
    ----------
    net : object
        GaussianNetwork object
    delta: int,
        Lattice spacing

    """

    dists = np.zeros((net.N**2))
    nshared = np.zeros((net.N**2))
    av, bv = np.meshgrid(np.arange(net.N),np.arange(net.N)) #Get all pairs
    A, B = av.ravel(), bv.ravel()
    for i in range(A.shape[0]):
        r1 = (net.X[A[i]],net.Y[A[i]])
        r2 = (net.X[B[i]], net.Y[B[i]])
        dists[i] = tordist(r1, r2, net.M, net.delta)
        nshared[i] = np.sum(np.logical_and(net.C[A[i],:], net.C[B[i],:]))

    unique = np.unique(dists)[1:]
    avgs_arr = np.zeros_like(unique)
    for i, val in enumerate(unique):
        idx = np.where(dists == val)
        avgs_arr[i] = np.mean(nshared[idx])

    return unique, avgs_arr

"""
Excitatory-Inhibitory Gaussian network
"""

def exin_net_avg_edeg(N, sigma_e, sigma_i, q, p_e, delta=1):

    """
    Average number of excitatory connections and inhibitory connections coming
    into and going out of an excitatory neuron for a single set of params
    (sigma_e, sigma_i, q, p_e)

    Parameters
    ----------
    N : int
        Number of neurons in the network. Must be a perfect square
    sigma_e : ndarray
        Standard deviation of the excitatory kernel
    sigma_i : ndarray
        Standard deviation of the inhibitory kernel
    q : ndarray
        Value of the sparsity parameter
    p_e : float,
        Fraction of neurons that are excitatory
    delta: int,
        Lattice spacing

    """

    p_i = 1-p_e
    M = int(round(np.sqrt(N)))
    xv, yv = np.meshgrid(np.arange(M),np.arange(M))
    X, Y = xv.ravel(), yv.ravel()
    dr_ij_vec = np.array([tordist((0,0),(X[i],Y[i]),M,delta) for i in range(N)])[1:]
    p_ee_in_vec = np.zeros_like(dr_ij_vec)
    p_ee_out_vec = np.zeros_like(dr_ij_vec)
    p_ei_in_vec = np.zeros_like(dr_ij_vec)
    p_ei_out_vec = np.zeros_like(dr_ij_vec)

    for i, dr_ij in enumerate(dr_ij_vec):
        k_ex_out = delta_gauss(dr_ij, sigma_e, delta)
        k_inh_in = delta_gauss(dr_ij, sigma_i, delta)
        p_ee_out, p_ee_in, p_ee_x  = trinomial(k_ex_out, k_ex_out, q)
        p_ei_out, p_ei_in, p_ei_x  = trinomial(k_ex_out, k_inh_in, q)
        p_ee_in_vec[i] = p_ee_in #E <- E
        p_ee_out_vec[i] = p_ee_out #E -> E
        p_ei_in_vec[i] = p_ei_in #E <- I
        p_ei_out_vec[i] = p_ei_out #E -> I

    avg_ee_in = np.sum(p_ee_in_vec*p_e)
    avg_ee_out = np.sum(p_ee_out_vec*p_e)
    avg_ei_in = np.sum(p_ei_in_vec*p_i)
    avg_ei_out = np.sum(p_ei_out_vec*p_e)

    return avg_ee_out, avg_ee_in, avg_ei_out, avg_ei_in

def exin_net_avg_ideg(N, sigma_e, sigma_i, q, p_e, delta=1):

    """
    Average number of excitatory connections and inhibitory connections coming
    into and going out of an inhibitory neuron for a single set of params
    (sigma_e, sigma_i, q, p_e)

    Parameters
    ----------
    N : int
        Number of neurons in the network. Must be a perfect square
    sigma_e : ndarray
        Standard deviation of the excitatory kernel
    sigma_i : ndarray
        Standard deviation of the inhibitory kernel
    q : ndarray
        Value of the sparsity parameter
    p_e : float,
        Fraction of neurons that are excitatory
    delta: int,
        Lattice spacing

    """

    p_i = 1-p_e
    M = int(round(np.sqrt(N)))
    xv, yv = np.meshgrid(np.arange(M),np.arange(M))
    X, Y = xv.ravel(), yv.ravel()
    dr_ij_vec = np.array([tordist((0,0),(X[i],Y[i]),M,delta) for i in range(N)])[1:]
    p_ii_in_vec = np.zeros_like(dr_ij_vec)
    p_ii_out_vec = np.zeros_like(dr_ij_vec)
    p_ie_in_vec = np.zeros_like(dr_ij_vec)
    p_ie_out_vec = np.zeros_like(dr_ij_vec)

    for i, dr_ij in enumerate(dr_ij_vec):
        k_inh_out = delta_gauss(dr_ij, sigma_i, delta)
        k_ex_in = delta_gauss(dr_ij, sigma_e, delta)
        p_ii_out, p_ii_in, p_ii_x  = trinomial(k_inh_out, k_inh_out, q)
        p_ie_out, p_ie_in, p_ie_x  = trinomial(k_inh_out, k_ex_in, q)
        p_ii_in_vec[i] = p_ii_in #I <- I
        p_ii_out_vec[i] = p_ii_out #I -> I
        p_ie_in_vec[i] = p_ie_in #I <- E
        p_ie_out_vec[i] = p_ie_out #I -> E

    avg_ii_in = np.sum(p_ii_in_vec*p_i)
    avg_ii_out = np.sum(p_ii_out_vec*p_i)
    avg_ie_in = np.sum(p_ie_in_vec*p_e)
    avg_ie_out = np.sum(p_ie_out_vec*p_i)

    return avg_ii_out, avg_ii_in, avg_ie_out, avg_ie_in

def exin_net_avg_deg_exp(N, sigmas, q, p_e, delta=1):

    """
    Average number of excitatory connections and inhibitory connections coming
    into and going out of excitatory and inhbitory neurons (experimental solution)
    for a fixed parameter set

    Parameters
    ----------
    N : int
        Number of neurons in the network. Must be a perfect square
    sigmas : ndarray
        Values of sigma_e and sigma_i to use (for meshgrid)
    q : ndarray
        Value of the sparsity parameter
    p_e : float,
     Fraction of neurons that are excitatory
    delta: int,
        Lattice spacing

    """

    nsigma = sigmas.shape[0]
    xv, yv = np.meshgrid(np.arange(nsigma),np.arange(nsigma))
    X, Y = xv.ravel(), yv.ravel()
    sig_e_v, sig_i_v = np.meshgrid(sigmas,sigmas)
    sigma_e, sigma_i = sig_e_v.ravel(), sig_i_v.ravel()
    ee_mat = np.zeros((nsigma,nsigma))
    ii_mat = np.zeros((nsigma,nsigma))
    ei_mat = np.zeros((nsigma,nsigma))
    ie_mat = np.zeros((nsigma,nsigma))
    for i in range(sigma_e.shape[0]):
        net = ExInGaussianNetwork(N, sigma_e[i], sigma_i[i], q, p_e=p_e, delta=1)
        ei = np.sum(net.C[:,net.ex_idx][net.in_idx,:],axis=1) #e->i
        ie = np.sum(net.C[:,net.in_idx][net.ex_idx,:],axis=1) #i->e
        ee = np.sum(net.C[:,net.ex_idx][net.ex_idx,:],axis=1) #e->e
        ii = np.sum(net.C[:,net.in_idx][net.in_idx,:],axis=1) #i->i
        ei_mat[X[i],Y[i]] = np.mean(ei)/(N*(1-p_e))
        ie_mat[X[i],Y[i]] = np.mean(ie)/(N*p_e)
        ee_mat[X[i],Y[i]] = np.mean(ee)/(N*p_e)
        ii_mat[X[i],Y[i]] = np.mean(ii)/(N*(1-p_e))
    return ee_mat, ii_mat, ei_mat, ie_mat

def exin_net_avg_edeg_fixq(N, sigmas, q, p_e, delta=1):

    """
    Average number of excitatory connections and inhibitory connections coming
    into and going out of an excitatory neuron for a fixed sparsity param q and
    varying (sigma_e, sigma_i)

    Parameters
    ----------
    N : int
        Number of neurons in the network. Must be a perfect square
    sigma_e : ndarray
        Standard deviation of the excitatory kernel
    sigma_i : ndarray
        Standard deviation of the inhibitory kernel
    q : ndarray
        Value of the sparsity parameter
    p_e : float,
     Fraction of neurons that are excitatory
    delta: int,
        Lattice spacing

    """

    nsigma = sigmas.shape[0]
    xv, yv = np.meshgrid(np.arange(nsigma),np.arange(nsigma))
    X, Y = xv.ravel(), yv.ravel()

    sig_e_v, sig_i_v = np.meshgrid(sigmas,sigmas)
    sigma_e, sigma_i = sig_e_v.ravel(), sig_i_v.ravel()
    n_ee_out = np.zeros((nsigma,nsigma))
    n_ee_in = np.zeros((nsigma,nsigma))
    n_ei_out = np.zeros((nsigma,nsigma))
    n_ei_in = np.zeros((nsigma,nsigma))
    for i in range(sigma_e.shape[0]):
        avg_ee_out, avg_ee_in, avg_ei_out, avg_ei_in =\
        exin_net_avg_edeg(N, sigma_e[i], sigma_i[i], q, p_e)
        n_ee_out[X[i],Y[i]] = avg_ee_out
        n_ee_in[X[i],Y[i]] = avg_ee_in
        n_ei_out[X[i],Y[i]] = avg_ei_out
        n_ei_in[X[i],Y[i]] = avg_ei_in
    return n_ee_out, n_ee_in, n_ei_out, n_ei_in

def exin_net_avg_ideg_fixq(N, sigmas, q, p_e, delta=1):

    """
    Average number of excitatory connections and inhibitory connections coming
    into and going out of an inhibitory neuron for a fixed sparsity param q and
    varying (sigma_e, sigma_i)

    Parameters
    ----------
    N : int
        Number of neurons in the network. Must be a perfect square
    sigma_e : ndarray
        Standard deviation of the excitatory kernel
    sigma_i : ndarray
        Standard deviation of the inhibitory kernel
    q : ndarray
        Value of the sparsity parameter
    p_e : float,
     Fraction of neurons that are excitatory
    delta: int,
        Lattice spacing

    """

    nsigma = sigmas.shape[0]
    xv, yv = np.meshgrid(np.arange(nsigma),np.arange(nsigma))
    X, Y = xv.ravel(), yv.ravel()

    sig_e_v, sig_i_v = np.meshgrid(sigmas,sigmas)
    sigma_e, sigma_i = sig_e_v.ravel(), sig_i_v.ravel()
    n_ii_out = np.zeros((nsigma,nsigma))
    n_ii_in = np.zeros((nsigma,nsigma))
    n_ie_out = np.zeros((nsigma,nsigma))
    n_ie_in = np.zeros((nsigma,nsigma))
    for i in range(sigma_e.shape[0]):
        avg_ii_out, avg_ii_in, avg_ie_out, avg_ie_in =\
        exin_net_avg_ideg(N, sigma_e[i], sigma_i[i], q, p_e)
        n_ii_out[X[i],Y[i]] = avg_ii_out
        n_ii_in[X[i],Y[i]] = avg_ii_in
        n_ie_out[X[i],Y[i]] = avg_ie_out
        n_ie_in[X[i],Y[i]] = avg_ie_in
    return n_ii_out, n_ii_in, n_ie_out, n_ie_in

def exin_net_shared_exp(C, a_idx, b_idx, M, delta=1):


    """
    This problem is hard and expensive. As of now, it can only be solved
    experimentally

    Given indices of presynaptic neurons of type A and indices of postsynaptic
    neuron of type B, find the average number of shared type A connections between
    type B neurons as a function of their distance from each other

    Parameters
    ----------
    C : ndarray
        Connectivity matrix
    a_idx : ndarray
        Indices of neurons of type A
    b_idx : ndarray
        Indices of neurons of type B
    M : ndarray
        Axial dimension of the lattice
    delta: int,
        Lattice spacing

    """


    def get_unique(dists, shared):
        unique = np.unique(dists)[1:]
        avgs_arr = np.zeros_like(unique)
        for i, val in enumerate(unique):
            idx = np.where(dists == val)
            avgs_arr[i] = np.mean(shared[idx])
        return unique, avgs_arr

    n = b_idx.shape[0]
    dists = np.zeros((n**2,))
    nshared = np.zeros((n**2,))

    xv, yv = np.meshgrid(np.arange(M),np.arange(M))
    X, Y = xv.ravel(), yv.ravel()

    #Get all possible pairs of excitatory neurons
    av, bv = np.meshgrid(b_idx,b_idx)
    A, B = av.ravel(), bv.ravel()
    for i in range(A.shape[0]):
        r1 = (X[A[i]],Y[A[i]])
        r2 = (X[B[i]], Y[B[i]])
        dists[i] = tordist(r1, r2, M, delta)
        nshared[i] = np.sum(np.logical_and(C[A[i],a_idx], C[B[i],a_idx]))

    dists, nshared = get_unique(dists,nshared)

    return dists, nshared
