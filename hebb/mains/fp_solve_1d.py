import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from hebb.util import *
from hebb.models import *

n_in = 100
nsteps = 1000
batches = 1

spike_gen = Poisson(n_in, nsteps)
spikes = spike_gen.run_generator()

fp = FokkerPlanck1D(n_in, nsteps=nsteps, dt=dt, batches=batches)
fp.solve(spike_gen.rates)

# #pdf attribute has shape (1, voltage, time)
# plt.plot(fp.bins, fp.pdf[0,:,0])
# plt.show()
