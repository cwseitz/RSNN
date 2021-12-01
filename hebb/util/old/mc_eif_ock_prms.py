import json
import numpy as np
from hebb.models import *

##################################################
## Set params for a Monte Carlo simulation for
## excitatory-inhibitory random network
##################################################
## Author: Clayton Seitz
## Copyright: 2021, The Hebb Project
## Email: cwseitz@uchicago.edu
##################################################

save_dir = '/home/cwseitz/Desktop/data/'

########################
## Define network params
########################

trials = 1
q = 0.5
dt = 0.1
N = 1000
Nrecord = 100
T = 10000
Nt = int(round(T/dt))
N_e = int(round(q*N))
N_i = int(round((1-q)*N))
maxns = N*T*0.02

########################
## Generate adj matrix
########################

p_ee = 0.15
pref = p_ee*np.ones((N,N))
net = np.random.uniform(0,1,size=(N,N))
net[net > pref] = 0
net[net != 0] = 1
np.fill_diagonal(net,0)

p0 = np.sum(net)/(N**2)
wmaxE = 5/(N*p0)
p_cond = wmaxE*0.25
net = p_cond*net


########################
## Define neuron params
########################

gl=[0.1, 0.1]
Cm=[1.0, 1.0]
vlb=[-100.0, -100.0]
vth=[30.0 ,30.0]
DeltaT=[1.4, 1.4]
vT=[-48.0, -48.0]
vl=[-72.0, -72.0]
vre=[-72.0, -72.0]
tref=[2.0, 2.0]
tausyne=5
tausyni=5
tausynx=5

########################
## Define stim params
########################

taux = 40
mxe = 1
mxi = 1
vxe = 81
vxi = 81

mxe0=(mxe/np.sqrt(N))
mxi0=(mxi/np.sqrt(N))

########################
## Define corr params
########################

Tmax = 0.1 #max time lag for cross-correlations (s))
u1 = 1
df = 50
fmax = 1/(2*dt*1e-3) #nyquist (folding) frequency (Hz)
freq = np.arange(0, 2*fmax, df)
# idx = np.argwhere(np.abs(freq) < 1e-6)
# freq[idx] = 1e-6
freq = freq.tolist()
nfreq = len(freq)

########################
## Dump params to disk
########################

params = {
'trials':trials,
'q':q,
'dt':dt,
'N':N,
'Nrecord':Nrecord,
'T':T,
'Nt':Nt,
'N_e':N_e,
'N_i':N_i,
'maxns':maxns,
'gl':gl,
'Cm':Cm,
'vlb':vlb,
'vth':vth,
'vl':vl,
'DeltaT':DeltaT,
'vT':vT,
'vre':vre,
'tref':tref,
'tausyne':tausyne,
'tausyni':tausyni,
'tausynx':tausynx,
'taux':taux,
'mxe':mxe,
'mxi':mxi,
'vxe':vxe,
'vxi':vxi,
'mxe0':mxe0,
'mxi0':mxi0,
'Tmax':Tmax,
'df':df,
'freq':freq,
'nfreq':nfreq,
'fmax':fmax

}

np.savez_compressed(save_dir + 'mc_eif_rand_weights', net)
with open(save_dir + 'params.json', 'w') as fp:
    json.dump(params, fp)
