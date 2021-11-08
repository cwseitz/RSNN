import numpy as np
import matplotlib.pyplot as plt
from hebb.models import *
from hebb.util import *

##################################################
## Main script for simulating an excitatory
## inhibitory random network
##################################################
## Author: Clayton Seitz
## Copyright: 2021, The Hebb Project
## Email: cwseitz@uchicago.edu
##################################################

##########
## Params
##########

#Time stuff
trials = 1
dt = 0.001 #sec
T = 1 #sec
nsteps = 1 + int(round(T/dt))

#Neuron parameters
tau_ref = 0.02 #sec
tau=0.02 #sec
v0=0 #mv
thr=10 #mv

#Network parameters
N=4000
p_e=0.5
j_ee=12.5
j_ei=20
j_ie=50
j_ii=50

p_ee=0.25
p_ei=0.25
p_ie=0.25
p_ii=0.25

w_ee=j_ee*p_ee*p_e
w_ei=j_ei*p_ei*p_e
w_ie=j_ie*p_ie*(1-p_e)
w_ii=j_ii*p_ii*(1-p_e)

#Timescale, mean and variance of ffwd input
mxe=0.015*np.sqrt(N)
mxi=0.01*np.sqrt(N)
vxe=0.05
vxi=0.05

#Check mean field firing rates
r_e = (mxe*j_ii - mxi*j_ie)/(j_ei*j_ie - j_ee*j_ii)
r_i = (mxe*j_ei - mxi*j_ee)/(j_ei*j_ie - j_ee*j_ii)
print(f'MFT: Excitatory rate: {r_e}, Inhibitory rate: {r_i}')

##########
## Objects
##########

print("Generating connectivity...")
J_xx = [j_ee,j_ei,-j_ie,-j_ii]/np.sqrt(N)
p_xx = [p_ee,p_ei,p_ie,p_ii]
net = ExInFixedNetwork(N, p_e, p_xx, J_xx)
net.make_weighted()
print("Done")

print("Running sim...")
ffwd = np.random.normal(mxe, vxe, size=(N, trials, nsteps))
rnn = LIF(T, dt, tau_ref, v0, net.C, trials, tau, thr, dtype=np.float16)
rnn.call(ffwd)
print("Done")

# ##########
# ## Plots
# ##########

fig_7(ffwd,net,rnn)
# fig_8(ffwd,net,rnn)
# fig_9(ffwd,net,rnn)
plt.show()
