import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from hebb.util import *
from hebb.models import *

n_in = 20
n_rec = 100
p_e = 1.0
p_xx = np.array([[0.2,0.2],
                 [0.2,0.2]])
nsteps = 100
dt = 1
batches = 50

spikes = poisson_input(nsteps, dt, units=n_in, batches=batches)
net = ExInLIF(n_in, n_rec, p_xx, p_e=p_e, batches=batches, period=nsteps)
v,z,r = net.call(spikes)

fig, ax = plt.subplots(3,1, sharex=True, sharey=True)
ax[0].imshow(v[:,0,:], cmap='gray')
ax[1].imshow(z[:,0,:], cmap='gray')
ax[2].imshow(r[:,0,:], cmap='gray')
plt.show()
