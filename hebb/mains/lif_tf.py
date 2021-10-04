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


#Trials & Timing
trials = 1 #number of trials
dt = 0.001 #1ms
T =  1.0 #100ms
tau_ref = 0.02 #3ms

#Network
N = 100
J = np.zeros((N,N))

currents = Poisson(T,dt,n_in,trials=trials).to_currents(f.XIJ)
lif = LIF(T, dt, tau_ref, J, trials=trials)

# sc = np.zeros_like(currents)
# sigma = 0.02
# iters = 10
# taus = np.linspace(dt, 100*dt, 100)
# rates = np.zeros((iters,len(taus),len(currents)))
#
# for i in range(iters):
#     for j, tau in enumerate(taus):
#         for k, current in enumerate(currents):
#             input = current*np.ones((N, batches, int(round(T/dt))+1))
#             input += sigma*np.random.normal(0,1,size=input.shape)
#             lif = LIF(T, dt, tau_ref, N=N, tau=tau, input=input, batches=batches)
#             lif.call()
#             rates[i,j,k] = np.sum(lif.Z[0,0,:])/T
#
# plt.imshow(np.mean(rates, axis=0), cmap='coolwarm')
# plt.xlabel('I')
# plt.ylabel('$\tau$')
# plt.show()
