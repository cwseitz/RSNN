import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from hebb.util import *
from hebb.models import *

##################################################
## Main script for simulating Brownian motion
## i.e. a Wiener Process
##################################################
## Author: Clayton Seitz
## Copyright: 2021, The Hebb Project
## Email: cwseitz@uchicago.edu
##################################################

nsteps = 1000
dt = 0.001
V0 = 5
beta = 0.2
batch_size = 1000

S = Brownian(V0, nsteps, dt, beta, batch_size=batch_size)
S.forward()

"""
Distribution of V in time
"""

colormap = cm.get_cmap('coolwarm')
colors = colormap(np.linspace(0, 1, nsteps))
norm = mpl.colors.Normalize(vmin=0, vmax=nsteps)

fig, ax = plt.subplots()

for i in range(1, nsteps):
    vals, bins = np.histogram(S.V[i,:], density=True)
    center = (bins[:-1] + bins[1:]) / 2
    ax.plot(center, vals, color=colors[i])

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
