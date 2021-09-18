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

S = StationaryOU(nsteps, tau, sigma, batch_size=batch_size, dv=0.1, v_max=v_max)
S.forward()
S.solve_fp_analytic()
S.histogram()

"""
Probability Densities
"""

steps = [100, 200, 999]

fig, ax = plt.subplots()

ax.plot(S._V, S.P_A[:,steps[0]], color='red', label='0ms', linestyle='--')
ax.plot(S._V, S.P_A[:,steps[1]], color='blue', label='100ms', linestyle='--')
ax.plot(S._V, S.P_A[:,steps[2]], color='cyan', label='200ms', linestyle='--')

ax.plot(S._V, S.P_S[:,steps[0]], color='red', label='0ms')
ax.plot(S._V, S.P_S[:,steps[1]], color='blue', label='100ms')
ax.plot(S._V, S.P_S[:,steps[2]], color='cyan', label='200ms')

ax.set_xlim([-v_max, v_max])
ax.set_xlabel('V')
ax.set_ylabel('P(V)')
plt.tight_layout()
plt.grid()

"""
Trajectories
"""

fig, ax = plt.subplots()

for i in range(batch_size):
    ax.plot(S.V[:,i], color='blue', alpha=0.3)

ax.set_xlabel('Time')
ax.set_ylabel('V')

plt.tight_layout()
plt.grid()
plt.show()
