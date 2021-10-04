import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from hebb.util import *
from hebb.models import *

##################################################
## Main script for simulating an Ornstein-Uhlenbeck
## process with non-stationary Gaussian noise
##################################################
## Author: Clayton Seitz
## Copyright: 2021, The Hebb Project
## Email: cwseitz@uchicago.edu
##################################################


nsteps = 1000
dt = 0.002
V_R = 0
tau = 0.1
f = 1000*dt
mu = 0.01*np.sin(2*np.pi*f*np.arange(nsteps)*dt)
sigma = 0.3
batch_size = 1000
xmin=0
xmax=2

S = NonStationaryOU(nsteps, V_R, tau, mu, sigma, batch_size=batch_size, xmin=xmin, xmax=xmax)
S.forward()

"""
Mean of drifting noise - a function of time
"""

fig, ax = plt.subplots()
ax.plot(np.arange(nsteps)*dt, mu, color='blue')


"""
Distribution of V in time - simulation
"""

colormap = cm.get_cmap('coolwarm')
colors = colormap(np.linspace(0, 1, nsteps))
norm = mpl.colors.Normalize(vmin=0, vmax=nsteps)

fig, ax = plt.subplots()

for i in range(1, nsteps):
    vals, bins = np.histogram(S.V[i,:], density=True)
    ax.plot(bins[:-1], vals, color=colors[i])

ax.set_xlim([xmin, xmax])
ax.set_xlabel('V')
ax.set_ylabel('P(V)')

fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=colormap), label='Time')
plt.tight_layout()
plt.grid()

"""
Trajectories
"""

colormap = cm.get_cmap('Blues')
colors = colormap(np.linspace(0, 1, batch_size))
norm = mpl.colors.Normalize(vmin=0, vmax=batch_size)

fig, ax = plt.subplots()

for i in range(batch_size):
    ax.plot(S.V[:,i], color='blue', alpha=0.3)

ax.set_xlabel('Time')
ax.set_ylabel('V')

plt.tight_layout()
plt.grid()

plt.show()
