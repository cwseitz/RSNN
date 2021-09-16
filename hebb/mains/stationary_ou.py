import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from hebb.util import *
from hebb.models import *

nsteps = 1000
dt = 0.002
tau = 0.1
sigma = 1
batch_size = 1000
v_max = 1

S = StationaryOU(nsteps, tau, sigma, batch_size=batch_size, v_max=v_max)
S.forward()
S.solve_fp_analytic()
S.solve_fp_numeric()

"""
Analytical solution
"""

colormap = cm.get_cmap('coolwarm')
colors = colormap(np.linspace(0, 1, nsteps))
norm = mpl.colors.Normalize(vmin=0, vmax=nsteps)


fig, ax = plt.subplots()
fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=colormap), label='Time')

for i in range(1, nsteps):
    ax.plot(S._V, S.P_A[:,i], color=colors[i])

ax.set_xlabel('V')
ax.set_ylabel('P(V)')
plt.tight_layout()
plt.title('Analytical Solution')

"""
Numerical solution
"""


colormap = cm.get_cmap('coolwarm')
colors = colormap(np.linspace(0, 1, nsteps))
norm = mpl.colors.Normalize(vmin=0, vmax=nsteps)


fig, ax = plt.subplots()
fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=colormap), label='Time')

for i in range(1, nsteps):
    ax.plot(S._V, S.P_N[i,:], color=colors[i])

ax.set_xlabel('V')
ax.set_ylabel('P(V)')
plt.tight_layout()
plt.title('Numerical Solution')


"""
Simulation
"""

colormap = cm.get_cmap('coolwarm')
colors = colormap(np.linspace(0, 1, nsteps))
norm = mpl.colors.Normalize(vmin=0, vmax=nsteps)

fig, ax = plt.subplots()

for i in range(1, nsteps):
    vals, bins = np.histogram(S.V[i,:], density=True)
    center = (bins[:-1] + bins[1:])/2
    ax.plot(center, vals, color=colors[i])

ax.set_xlim([0, v_max])
ax.set_xlabel('V')
ax.set_ylabel('P(V)')

fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=colormap), label='Time')
plt.tight_layout()
plt.grid()

# """
# Trajectories
# """
#
# colormap = cm.get_cmap('Blues')
# colors = colormap(np.linspace(0, 1, batch_size))
# norm = mpl.colors.Normalize(vmin=0, vmax=batch_size)
#
# fig, ax = plt.subplots()
#
# for i in range(batch_size):
#     ax.plot(S.V[:,i], color='blue', alpha=0.3)
#
# ax.set_xlabel('Time')
# ax.set_ylabel('V')
#
# plt.tight_layout()
# plt.grid()

plt.show()
