import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from hebb.util import *
from hebb.models import *
from skimage.io import imsave

##################################################
## Main script for simulating a 2D lattice of neurons
## with spatially dependent recurrent connectivity
##################################################
## Author: Clayton Seitz
## Copyright: 2021, The Hebb Project
## Email: cwseitz@uchicago.edu
##################################################

N = 400
M = int(round(np.sqrt(N)))

#Trials & Timing
trials = 100 #number of trials
dt = 0.001 #1ms
T =  2.0 #100ms
tau_ref = 0.02
nsteps = 1 + int(round(T/dt))

#Connectivity parameters
sigma_e = 2
sigma_i = 2
q = 0.8
p_e = 0.8
J_ee, J_ei, J_ie, J_ii = [0.1,0.1,-0.1,-0.1]
net = ExInGaussianNetwork(N, sigma_e, sigma_i, q, p_e=p_e)
net.make_weighted(J_ee, J_ei, J_ie, J_ii)

#Stimulus
mu = 0.5*np.ones((N,))
var = 0.1**2
cov = np.eye(N)*var
currents = np.random.multivariate_normal(mu, cov, size=(nsteps, trials)).T

lif = LIF(T, dt, tau_ref, net.C, trials=trials)
lif.call(currents)

v_mov = np.zeros((nsteps,1,1,M,M))
X = np.mean(lif.I, axis=1)
for i in range(nsteps):
    frame = X[:,i]
    frame = frame.reshape((M,M))
    v_mov[i,0,0,:,:] = frame
imsave('../data/temp.tif', v_mov, metadata={'axes': 'TZCYX'})

fig_5(lif, net)
plt.show()
