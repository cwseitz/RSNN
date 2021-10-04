import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from hebb.util import *
from hebb.models import *

##################################################
## Main script for simulating a Brunel network (BN)
## The BN is an excitatory population of LIF neurons
## coupled to itself and an inhibitory population.
##################################################
## Author: Clayton Seitz
## Copyright: 2021, The Hebb Project
## Email: cwseitz@uchicago.edu
##################################################


n_excite = 80
n_inhib = 20
n_in = 10
N = n_excite + n_inhib

#Trials & Timing
trials = 1 #number of trials
dt = 0.001 #1ms
T =  1.0 #100ms
tau_ref = 0.002 #3ms

#Params and input
J_xx = [2, 2, -2, -2] #J_ee, J_ei, J_ie, J_ii
p = 0.2

#Network
f = BrunelNetwork(n_excite, n_inhib, n_in, p, J_xx)
f.run_generator()
f.plot()
currents = Poisson(T,dt,n_in,trials=trials).to_currents(f.XIJ)
lif = LIF(T, dt, tau_ref, f.CIJ, trials=trials)
lif.call(currents)
plot_activity(lif)

plt.show()
