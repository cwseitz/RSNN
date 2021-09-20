import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from hebb.util import *
from hebb.models import *

n_in = 100
n_rec = 100
p_e = 1.0
p_xx = np.array([[0.2,0.2],
                 [0.2,0.2]])
nsteps = 1000
dt = 1
batches = 100

spikes = Poisson(n_in, nsteps, dt=dt, batches=batches).run_generator()
net = ExInLIF(n_in, n_rec, p_xx, nsteps, tau=1, p_e=p_e, batches=batches)
v,z,r = net.call(spikes)

batch_ind = 0
fig, ax = plt.subplots()
ax.imshow(spikes[:,batch_ind,:], cmap='gray')

fig, ax = plt.subplots(3,1, sharex=True, sharey=True)

ax[0].imshow(v[:,batch_ind,:], cmap='gray')
ax[1].imshow(z[:,batch_ind,:], cmap='gray')
ax[2].imshow(r[:,batch_ind,:], cmap='gray')

plot_input_statistics(v, bins=10)
plot_voltage_statistics(v, bins=10)

plt.show()
