import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from hebb.util import *
from hebb.models import *

##################################################
## Main script for simulating an Ornstein-Uhlenbeck
## process with stationary Gaussian noise
##################################################
## Author: Clayton Seitz
## Copyright: 2021, The Hebb Project
## Email: cwseitz@uchicago.edu
##################################################

dt = 0.002
t = np.arange(0,2,dt)
tau = 0.1
sigma = 1
batch_size = 1000
v_max = 1

S = StationaryOU(t, tau, sigma, batch_size=batch_size, dv=0.1, v_max=v_max)
S.forward()
S.solve_fp_analytic()
S.histogram()

"""
Probability Densities
"""

steps = [100, 200, 999]

fig, ax = plt.subplots()

ax.plot(S._V, S.P_A[:,steps[0]], color='red', linestyle='--',)
ax.plot(S._V, S.P_A[:,steps[1]], color='blue', linestyle='--')
ax.plot(S._V, S.P_A[:,steps[2]], color='cyan', label='FP - 200ms', linestyle='--')

ax.plot(S._V, S.P_S[:,steps[0]], color='red')
ax.plot(S._V, S.P_S[:,steps[1]], color='blue')
ax.plot(S._V, S.P_S[:,steps[2]], color='cyan', label='Sim - 200ms')

ax.set_xlim([-v_max, v_max])
ax.set_xlabel('V', fontsize=14)
ax.set_ylabel('P(V)', fontsize=14)
plt.tight_layout()
plt.legend()
plt.grid()

"""
Trajectories
"""

fig, ax = plt.subplots()

for i in range(batch_size):
    ax.plot(t, S.V[:,i], color='salmon', alpha=0.3)

ax.set_xlabel('Time (s)', fontsize=14)
ax.set_ylabel('V', fontsize=14)

plt.tight_layout()
plt.grid()
plt.show()
