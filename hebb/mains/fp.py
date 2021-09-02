import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from hebb.util import *
from hebb.models import *

n_in = 100
n_rec = 100
p_e = 0.8
p_xx = np.array([[0.2,0.2],
                 [0.2,0.2]])
nsteps = 100
units = 100
dt = 1

net = ExInLIF(n_in, n_rec, p_xx, p_e=p_e)
net.in_cmg.run_generator()
net.rec_cmg.run_generator()
plt.imshow(net.rec_cmg.weights)
plt.show()


#rates = np.zeros((units, nsteps))
#rates[np.random.randint(0, units, size=10), :] = 0.1

# colors = cm.coolwarm(np.linspace(0, 1, nsteps))
# fig, ax = plt.subplots(1,3)
#
# nsims = 1000
# spike_tensor = []
# for i in range(nsims):
#     spikes = poisson_input(nsteps, dt, units=units, rates=None)
#     spike_tensor.append(spikes)
# spike_tensor = np.array(spike_tensor)
# input = np.sum(spike_tensor, axis=1)
#
# ax[0].imshow(spike_tensor[0], cmap='gray')
#
# for i in range(input.shape[-1]):
#     ax[1].plot(input[i])
#
# for i in range(input.shape[-1]):
#     vals, bins = np.histogram(input[:, i])
#     ax[2].plot(bins[:-1], vals, color=colors[i])
#
# plt.tight_layout()
# plt.show()
