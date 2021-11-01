import numpy as np
from ..models import *
from .math import *

##################################################
## Library of experimental solutions to problems
## associated with Gaussian networks
##################################################
## Author: Clayton Seitz
## Copyright: 2021, The Hebb Project
## Email: cwseitz@uchicago.edu
##################################################


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
