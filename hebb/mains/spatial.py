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
N = 6400
p = 0.2
J_xx = [0.2, 0.2, -0.2, -0.2]
f = SpatialNetwork2D(N, p, J_xx, sigma_e=50, sigma_i=50, alpha=500)
f.plot()

# #Clamped LIF
# lif = ClampedLIF(T, dt, tau_ref, f.CIJ, trials=trials)
# shape = lif.shape
#
# #Poisson spikes for clamp
# r0 = 20 #Hz
# N,trials,steps = shape
# clamp = np.zeros((N,trials,steps))
# x = np.random.randint(0,N,500)
# clamp[x,:,:] = 1
# rates = clamp*r0
# poisson = Poisson(T,dt,N,trials=trials,rates=rates)
# spikes = poisson.run_generator()
#
# #Run the sim
# lif.call(spikes, clamp)
# plot_activity(lif)
# # unit_i_stats(lif)
# # pop_v_stats(lif)
# plot_unit(lif, unit=105)
plt.show()
