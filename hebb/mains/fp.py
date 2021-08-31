import numpy as np
import matplotlib.pyplot as plt
from hebb.util import *
from hebb.models import *

nsteps = 1000
units = 100
dt = 1

rates = np.zeros((units, nsteps))
rates[np.random.randint(0, units, size=10), :] = 0.1

spikes = poisson_input(nsteps, dt, rates=rates)

plt.imshow(spikes)
plt.show()

input = np.sum(spikes, axis=0)
input = np.cumsum(input)
plt.plot(input)
plt.show()
