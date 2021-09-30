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
mx_lvl = 10
E = 5
sz_cl = 9
f = FractalNetwork(mx_lvl, E, sz_cl)
J = f.run_generator(scale=True)

#Trials & Timing
trials = 100 #number of trials
dt = 0.001 #1ms
T =  1.0 #100ms
tau_ref = 0.003 #3ms

#LIF Network
lif = ClampedLIF(T, dt, tau_ref, J, trials=trials, thr=1.0)
shape = lif.shape

#Poisson spikes for clamp
r0 = 50 #Hz
N,trials,steps = shape
clamp = np.zeros((N,trials,steps))
clamp[2**sz_cl-1:,:,:] = 1
rates = clamp*r0
poisson = Poisson(T,dt,N,trials=trials,rates=rates)
spikes = poisson.run_generator()

#Run the sim
lif.call(spikes, clamp)
lif.plot_unit()
lif.plot_activity()
lif.plot_rate_hist()
plt.show()
