import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from hebb.util import *
from hebb.models import *

n_in = 20
n_rec = 60
p_e = 1.0
p_xx = np.array([[0.2,0.2],
                 [0.2,0.2]])
nsteps = 100
dt = 1

spikes = poisson_input(nsteps, dt, units=n_in, rates=None)
net = ExInLIF(n_in, n_rec, p_xx, p_e=p_e, period=nsteps)
v = net.call(spikes)
plt.imshow(v[:,0,:], cmap='gray')
plt.show()
