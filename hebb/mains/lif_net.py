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

mx_lvl = 8
E = 2
sz_cl = 5
N = 2**mx_lvl
dt = 0.001
t = np.arange(0, 1, dt)
batches = 1

cmg = InputConnectivityGenerator(N)
W = cmg.run_generator()
spikes = Poisson(t, N, batches=batches, random_select=200).run_generator()

f = FractalConnect(mx_lvl, E, sz_cl)
colors = ['cornflowerblue', 'salmon', 'black', 'gray']
J, k = f.run_generator()

f.plot(colors=colors)

lif = LIF(t, N, batches=batches, X=spikes, g_l=1, tau=1)
lif.W = W
lif.J = J
lif.plot_weights()
# lif.call()
# lif.plot_activity()
# lif.plot_input_stats()
plt.show()
