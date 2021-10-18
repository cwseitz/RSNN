import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from hebb.util import *
from hebb.models import *

##################################################
## Main script for simulating a 2D lattice of neurons
## with spatially dependent recurrent connectivity
##################################################
## Author: Clayton Seitz
## Copyright: 2021, The Hebb Project
## Email: cwseitz@uchicago.edu
##################################################

#Network
N = 525
rhos = [1, 5, 10]
colors=['red', 'blue', 'cyan']

for i, rho in enumerate(rhos):
    net = HOGN(N, sigma=5, delta=1, rho=rho)
    sigmas = np.arange(5, 100, 0.5)
    avg_n_ij = hogn_avg_out_deg(net, sigmas)
    plt.plot(sigmas, avg_n_ij, color=colors[i])
plt.show()
