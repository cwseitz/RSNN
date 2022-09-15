import json
import numpy as np
from rsnn.models import *

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
q = 1.0
dt = 0.1
N = 1000
Nrecord = 100
T = 10000
Nt = int(round(T/dt))
N_e = int(round(q*N))
N_i = int(round((1-q)*N))
maxns = N*T*0.1

########################
## Generate adj matrix
########################

net = np.zeros((N,N))

########################
## Define neuron params
########################

gl=[1/10, 1/10]
Cm=[1.0, 1.0]
vlb=[-100.0, -100.0]
vth=[-10.0 ,-10.0]
DeltaT=[2.0, 2.0]
vT=[-50.0, -50.0]
vl=[-60.0, -60.0]
vre=[-65.0, -65.0]
tref=[1.5, 1.5]
tausyne=4.0
tausyni=4.0
tausynx=4.0

########################
## Define stim params
########################

taux = 40
mxe = mxi = 2
vxe = vxi = 81
mxe0 = mxe
mxi0 = mxi

########################
## Define corr params
########################

Tmax = 50 #max time lag for cross-correlations (ms)
cc_dt = 1 #resolution for cross correlations (ms)
df = 1/(2*Tmax) #frequency resolution
fmax = 1/(2*cc_dt)
freq = np.arange(-fmax, fmax, df)
idx = np.argwhere(np.abs(freq) < 1e-6)
freq[idx] = 1e-6
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

np.savez_compressed(save_dir + 'mc_eif_ucpld_weights', net)
with open(save_dir + 'params.json', 'w') as fp:
    json.dump(params, fp)
