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

#Trials & Timing
trials = 1 #number of trials
dt = 0.001 #1ms
T =  2.0 #100ms
tau_ref = 0.002 #3ms

#Network
N = 1600
p = 0.2
J_xx = [0.2, 0.2, -0.2, -0.2]
f = SpatialNetwork2D(N, p, J_xx, sigma_e=2, sigma_i=2, alpha=10)
f.make_grid()
f.pairwise_stats(1, 100, 1, 500, rho_1=0.75, rho_2=0.75)
fig_1(f)
plt.show()
