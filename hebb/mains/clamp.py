import numpy as np
from hebb.models import *
from hebb.util import *

##################################################
## Main script for simulating a network of LIF
## neurons whe
##################################################
## Author: Clayton Seitz
## Copyright: 2021, The Hebb Project
## Email: cwseitz@uchicago.edu
##################################################

#Connectivity Matrix
mx_lvl = 6
E = 3
sz_cl = 5
f = FractalNetwork(mx_lvl, E, sz_cl)
J = 0.03*f.run_generator()

#Trials & Timing
trials = 200 #number of trials
dt = 0.001 #1ms
T =  1.0 #100ms
tau_ref = 0.003 #3ms

#LIF Network
lif = ClampedLIF(T, dt, tau_ref, J, trials=trials)
shape = lif.shape

#Poisson spikes for clamp
r0 = 20 #Hz
N,trials,steps = shape
clamp = np.zeros((N,trials,steps))
clamp[2**sz_cl-1:,:,:] = 1
rates = clamp*r0
poisson = Poisson(T,dt,N,trials=trials,rates=rates)
spikes = poisson.run_generator()

#Run the sim
lif.call(spikes, clamp)
lif.plot_activity()
plt.show()
