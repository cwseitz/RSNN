import numpy as np
from hebb.models import *

##################################################
## Main script for computing the transfer function
## (f-I) curve for a LIF neuron for different
## refractory period and time constant combinations
##################################################
## Author: Clayton Seitz
## Copyright: 2021, The Hebb Project
## Email: cwseitz@uchicago.edu
##################################################

batches = 1 #number of trials
N = 1 #number of neurons

#Timing
dt = 0.001 #1ms
T =  0.1 #100ms
tau_ref = 0.003 #3ms
dI = 0.0004
currents = np.arange(-0.04,0.04,dI)
sc = np.zeros_like(currents)
sigma = 0.02
iters = 20
rates = np.zeros((iters,len(currents)))

for i in range(iters):
    for j, current in enumerate(currents):
        input = current*np.ones((N, batches, int(round(T/dt))+1))
        input += sigma*np.random.normal(0,1,size=input.shape)
        lif = LIF(T, dt, tau_ref, N=N, tau=100*dt, input=input, batches=batches)
        lif.call()
        rates[i,j] = np.sum(lif.Z[0,0,:])/T

plt.scatter(currents, np.mean(rates, axis=0), color='black', marker='x')
plt.show()
