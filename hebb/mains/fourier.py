import numpy as np
import matplotlib.pyplot as plt
from hebb.models import *
from hebb.util import *

N = 1600 #total number of neurons
M = 1500 #number of neurons clamped

#Trials & Timing
trials = 5 #number of trials
dt = 0.001 #1ms
T =  1.0 #100ms
tau_ref = 0.02 #3ms
nsteps = 1 + int(round(T/dt))

#Generate the clamp
clamped_idx = np.random.choice(N, size=(M,), replace=False)
clamp = np.zeros((N,1))
clamp[clamped_idx] = 1

#Generate input spikes
rates = np.ones((N, trials, nsteps))
t = np.arange(0, nsteps, 1)*dt
f = 5
w = np.ones((N,))*2*np.pi*f
phi = np.pi/4
s = 50*np.sin(np.outer(w,t) + phi)**2
rates = np.einsum('ijk,ik -> ijk', rates, s)
rates = np.einsum('ijk,ik -> ijk', rates, clamp)
poisson = Poisson(T,dt,N,rates=rates, trials=trials)
poisson.run_generator()

#Initialize network
p = 0.2
J_xx = [1, 1, -1, -1]
f = SpatialNetwork2D(N, p, J_xx, sigma_e=2, sigma_i=2, alpha=10)

#Initialize neuron model
lif = ClampedLIF(T, dt, tau_ref, f.CIJ, trials=trials)
lif.call(poisson.spikes, clamped_idx)

#Figure
ax0, ax1, ax2, ax3, ax4, ax5 = fig_2()
focal = 10
add_spectral_graph(ax0, f.CIJ, f.in_idx)
add_raster(ax1, poisson.spikes, n_units=100)
add_activity(ax2, poisson.spikes)
add_unit_voltage(ax3, lif, unit=focal)
add_unit_current(ax4, lif, unit=focal)
add_unit_spikes(ax5, lif, unit=lif.no_clamp_idx[focal])
plt.show()
