import numpy as np
import scipy as p
import matplotlib.pyplot as plt
from hebb.util import *
from hebb.models import *

##################################################
## Main script for simulating a single Leaky
## Integrate and Fire (LIF) neuron model
##################################################
## Author: Clayton Seitz
## Copyright: 2021, The Hebb Project
## Email: cwseitz@uchicago.edu
##################################################

dt = 0.1
batches = 10
N = 100
t = np.arange(0.0, 100, dt)

#Generate input tensor (N, batches, time)
# input = np.zeros((N, batches, len(t)))
# input[:,:,1000:2000] = 0.2
# input[:,:,3000:4000] = 0.1

input = np.random.normal(2, 0.1, size=(N, batches, len(t)))

#Generate input spikes
# N_x = 100
# X = Poisson(t, N_x).run_generator()

lif = LIF(t, N=N, input=input, batches=batches)
lif.call()
lif.plot_unit()
plt.show()
