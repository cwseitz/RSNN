import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from hebb.util import *
from hebb.models import *

##################################################
## Main script for simulating a Brunel network (BN)
## The BN is an excitatory population of LIF neurons
## coupled to itself and an inhibitory population.
##################################################
## Author: Clayton Seitz
## Copyright: 2021, The Hebb Project
## Email: cwseitz@uchicago.edu
##################################################


n_excite = 800
n_inhib = 200
p_ee, p_ei, p_ie, p_ii = [0.5, 0.1, 0.1, 0.5]

dt = 0.001
t = np.arange(0, 1, dt)
batches = 1
mu = -0.64
sigma = 0.51

N = 1000
# cmg = InputConnectivityGenerator(N)
# W = cmg.run_generator()
# spikes = Poisson(t, N, batches=batches).run_generator()
# W.plot()

f = BrunelNetwork(n_excite, n_inhib, p_ee, p_ei, p_ie, p_ii, mu, sigma)
f.run_generator()
f.make_weighted()
f.plot()
plt.show()

# lif = LIF(t, N, batches=batches, X=spikes, g_l=1, tau=1)
# lif.W = W
# lif.J = f.CIJ
# lif.call()
# lif.plot_activity()
# lif.plot_unit()
# lif.plot_input_stats()
# plt.show()
