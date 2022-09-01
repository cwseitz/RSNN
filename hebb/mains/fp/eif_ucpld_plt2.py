import numpy as np
import matplotlib.pyplot as plt
import json
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from hebb.util import *

save_dir = '/home/cwseitz/Desktop/data/'
#######################################
## Load the parameters used
#######################################

with open(save_dir + 'params.json', 'r') as fh:
    params = json.load(fh)

#######################################
## Load the Monte-Carlo sim data
#######################################

Atfile = 'lr_eif_ucpld_At.npz'
At_arr = np.load(save_dir + Atfile)['arr_0']

#######################################
## Generate a figure for frequency response
#######################################

freq = np.abs(np.array(params['freq'])[:50])
mag = 1000*np.abs(At_arr)[:50,:50]
phi = np.angle(At_arr)[:50,:50]
M = 100
mu_arr = np.linspace(0,4,M)[:50]

xx, yy = np.meshgrid(freq, mu_arr)

fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1 = fig.gca(projection='3d')
surf1 = ax1.plot_surface(xx, yy, mag ,rstride=1, cstride=1, cmap=plt.cm.coolwarm,
        linewidth=0)

fig.colorbar(surf1, shrink=0.5, aspect=10, pad = 0, location='top', label='Firing Rate (Hz)')
ax1.view_init(azim=0, elev=90)
ax1.grid(False)
ax1.set_zticks([])
ax1.set_xlabel(r'$\omega\;[\mathrm{kHz}]$', labelpad=10)
ax1.set_ylabel(r'$\mu\; [\mu A/\mathrm{cm}^{2}]$', labelpad=10)

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2 = fig.gca(projection='3d')
surf2 = ax2.plot_surface(xx, yy, phi ,rstride=1, cstride=1, cmap=plt.cm.coolwarm,
        linewidth=0)

fig.colorbar(surf2, shrink=0.5, aspect=10, pad = 0, location='top', label='Phase (rad)')
ax2.view_init(azim=0, elev=90)
ax2.grid(False)
ax2.set_zticks([])
ax2.set_xlabel(r'$\omega\;[\mathrm{kHz}]$', labelpad=10)
ax2.set_ylabel(r'$\mu\; [\mu A/\mathrm{cm}^{2}]$', labelpad=10)


plt.show()
