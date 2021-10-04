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
mx_lvl = 8
E = 5
sz_cl = 7
J0 = 1 #mV
net = FractalNetwork(mx_lvl, E, sz_cl)
net.run_generator(scale=False)
net.J = J0*net.J
net.plot()

#Trials & Timing
trials = 1 #number of trials
dt = 0.001 #1ms
T =  0.5 #100ms
tau_ref = 0.002 #3ms

#LIF Network
lif = ClampedLIF(T, dt, tau_ref, net.J, trials=trials)
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
plot_activity(lif)
unit_i_stats(lif)
pop_v_stats(lif)
plot_unit(lif)
plt.show()
