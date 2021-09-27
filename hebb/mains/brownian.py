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

dt = 0.001
t = np.arange(0,1,dt)
V0 = 0
sigma = 0.2
batch_size = 1000
v_th = 0.6

S = Brownian(t, V0, sigma, batch_size=batch_size)
S.forward()

"""
Distribution of V in time
"""

colormap = cm.get_cmap('coolwarm')
colors = colormap(np.linspace(0, 1, len(t)))
norm = mpl.colors.Normalize(vmin=0, vmax=t.max())

fig, ax = plt.subplots()

for i in range(1, len(t)):
    vals, bins = np.histogram(S.V[i,:], density=True)
    center = (bins[:-1] + bins[1:]) / 2
    ax.plot(center, vals, color=colors[i])

ax.vlines(v_th, *ax.get_ylim(), color='black')
ax.set_xlabel('V', fontsize=14)
ax.set_ylabel('P(V)', fontsize=14)


fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=colormap), label='Time (s)')
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
    ax.plot(t, S.V[:,i], color='blue', alpha=0.3)
ax.set_title('$\sigma=0.2$ ($D=0.08$)')
ax.hlines(v_th, *ax.get_xlim(), color='black')

ax.set_xlabel('Time (s)', fontsize=14)
ax.set_ylabel('V', fontsize=14)

plt.tight_layout()
plt.grid()
plt.show()
