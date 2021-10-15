import numpy as np
from hebb.util import *
from hebb.models import *

##################################################
## Main script for simulating an Ornstein-Uhlenbeck
## process with non-stationary Gaussian noise
##################################################
## Author: Clayton Seitz
## Copyright: 2021, The Hebb Project
## Email: cwseitz@uchicago.edu
##################################################


T = 1.0 #sec
dt = 0.002 #sec
tau = 0.1 #sec
freq = 1000*dt
stim = 0.01*np.sin(2*np.pi*freq*np.arange(0,T,dt))**2
sigma = 1.0
trials = 3

ou = NonStationaryOU(T, dt, tau, stim, sigma, trials=trials, xmin=0, xmax=2)
ou.forward()

ou.plot_trajectories()
plt.show()
