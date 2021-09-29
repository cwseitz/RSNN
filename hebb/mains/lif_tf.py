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

#Input current
currents = np.linspace(-0.04,0.04,100)

sc = np.zeros_like(currents)
sigma = 0.02
iters = 10
taus = np.linspace(dt, 100*dt, 100)
rates = np.zeros((iters,len(taus),len(currents)))


for i in range(iters):
    for j, tau in enumerate(taus):
        for k, current in enumerate(currents):
            input = current*np.ones((N, batches, int(round(T/dt))+1))
            input += sigma*np.random.normal(0,1,size=input.shape)
            lif = LIF(T, dt, tau_ref, N=N, tau=tau, input=input, batches=batches)
            lif.call()
            rates[i,j,k] = np.sum(lif.Z[0,0,:])/T

plt.imshow(np.mean(rates, axis=0), cmap='coolwarm')
plt.xlabel('I')
plt.ylabel('$\tau$')
plt.show()

# plt.scatter(currents, np.mean(rates, axis=0), color='black', marker='x')
# plt.show()
