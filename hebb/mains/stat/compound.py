import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from hebb.util import *
from hebb.models import *

N = 5000
trials = 5 #number of trials
dt = 0.001 #1ms
T =  1.0 #100ms
nsteps = 1 + int(round(T/dt))
J = np.ones((N,))
rates = 20*np.ones((N,trials,nsteps))


cmpd = CompoundPoisson(T, dt, J, trials, rates=rates)
I = cmpd.run_generator()
plt.hist(I.flatten())
plt.show()
