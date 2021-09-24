import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from hebb.util import *
from hebb.models import *

##################################################
## Main script for simulating a network of
## Leaky Integrate and Fire (LIF) neurons
##################################################
## Author: Clayton Seitz
## Copyright: 2021, The Hebb Project
## Email: cwseitz@uchicago.edu
##################################################

mx_lvl = 8
E = 2
sz_cl = 7
N = 2**mx_lvl
dt = 0.001
t = np.arange(0, 1, dt)
batches = 1

cmg = InputConnectivityGenerator(N)
W = cmg.run_generator()
spikes = Poisson(t, N, batches=batches, random_select=200).run_generator()

f = FractalConnect(mx_lvl, E, sz_cl)
colors = ['cornflowerblue', 'salmon']
J, k = f.run_generator()
f.plot(colors=colors)

lif = LIF(t, N, batches=batches, X=spikes, g_l=1, tau=1)
lif.W = W
lif.J = J
lif.plot_weights()
lif.call()
lif.plot_activity()
lif.plot_input_stats()
plt.show()
