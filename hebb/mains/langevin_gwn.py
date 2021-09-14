import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from hebb.util import *
from hebb.models import *

nsteps = 100
V_0 = 0
J = 1
alpha = 0.01
tau = 1
batch_size = 1000

langGWN = LangevinGWN(nsteps, V_0, J, alpha, batch_size=batch_size, tau=tau)
langGWN.forward()

"""
Distribution of V in time
"""

colormap = cm.get_cmap('coolwarm')
colors = colormap(np.linspace(0, 1, nsteps))
norm = mpl.colors.Normalize(vmin=0, vmax=nsteps)

fig, ax = plt.subplots()

for i in range(nsteps):
    vals, bins = np.histogram(langGWN.V[i,:])
    ax.plot(bins[:-1], vals, color=colors[i])

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
    ax.plot(langGWN.V[:,i], color=colors[i], alpha=0.3)

ax.set_xlabel('Time')
ax.set_ylabel('V')

fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=colormap), label='Simulation Index')
plt.tight_layout()
plt.grid()
plt.show()
