import numpy as np
import scipy as p
import matplotlib.pyplot as plt
from hebb.util import *
from hebb.models import *

##################################################
## Main script for testing a single Leaky
## Integrate and Fire (LIF) neuron model
##################################################
## Author: Clayton Seitz
## Copyright: 2021, The Hebb Project
## Email: cwseitz@uchicago.edu
##################################################

batches = 1 #number of trials
N = 1 #number of neurons

dt = 0.001 #1ms
T = 0.02 #10ms
tau_ref = 0.003 #3ms

#Generate input tensor (N, batches, time)
input = np.zeros((N, batches, int(round(T/dt))+1))
input[:,:,5:12] = 0.7

lif = LIF(T, dt, tau_ref, N=N, input=input, batches=batches)
lif.call()
lif.plot_unit()
plt.show()
