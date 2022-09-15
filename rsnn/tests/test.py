import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from hebb.models import *
from hebb.util import *

def test_trinomial():

    """
    Test the vectorized trinomial sampling functions used to generate synapses

    Parameters
    ----------
    """

    start_time = time.time()
    #test with scalar arrays
    size = ()
    a = np.random.uniform(0,1,size=size)
    b = np.random.uniform(0,1,size=size)
    c = np.random.uniform(0,1,size=size)
    samp = sample_trinomial(a,b,c)

    #test with vector arrays
    size = (100,)
    a = np.random.uniform(0,1,size=size)
    b = np.random.uniform(0,1,size=size)
    c = np.random.uniform(0,1,size=size)
    samp = sample_trinomial(a,b,c)

    #test with matrix arrays
    size=(100,100)
    a = np.random.uniform(0,1,size=size)
    b = np.random.uniform(0,1,size=size)
    c = np.random.uniform(0,1,size=size)
    samp = sample_trinomial(a,b,c)
    print("--- %s seconds ---" % (time.time() - start_time))


def test_exin_gauss(N=2500,sigma_e=4,sigma_i=2,q0=0.8,p_e=0.8,delta=1):

    """
    Test the generation of excitatory-inhibitory gaussian networks

    Parameters
    ----------
    """

    times = []
    N_arr = np.arange(80,130,10)**2
    for N in N_arr:
        start_time = time.time()
        exin = ExInGaussianNetwork(N, sigma_e, sigma_i, q0, p_e=p_e, delta=delta)
        print(f'N = {N}, Size: {sys.getsizeof(exin.C)} bytes')
        print("--- %s seconds ---" % (time.time() - start_time))
        times.append(time.time() - start_time)
        del exin
        exin = ExInGaussianNetwork_Sparse(N, sigma_e, sigma_i, q0, p_e=p_e, delta=delta)
        print(f'N = {N}, Size: {sys.getsizeof(exin.C)} bytes')
        print("--- %s seconds ---" % (time.time() - start_time))
        times.append(time.time() - start_time)
        del exin
    times = np.array(times)
