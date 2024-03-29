import numpy as np
import matplotlib.pyplot as plt
import json
from rsnn.models import *
from rsnn import _rsnn

##################################################
## Solve the Fokker-Planck equation for EIF 
## model in the steady state, driven by
## additive Gaussian white noise
##################################################
## Author: Clayton Seitz
## Copyright: 2021, The Hebb Project
## Email: cwseitz@uchicago.edu
##################################################

save_dir = '/home/cwseitz/Desktop/data/'

#######################################
## Load the parameters used
#######################################

with open(save_dir + 'params.json', 'r') as fh:
    params = json.load(fh)

N = params['N']
N_e = params['N_e']
N_i = params['N_i']
gl = params['gl'][0]
Cm = params['Cm'][0]
DeltaT = params['DeltaT'][0]
vT = params['vT'][0]
vth = params['vth'][0]
vlb = params['vlb'][0]
vre = params['vre'][0]
vl = params['vl'][0]
tref = params['tref'][0]
dV = 0.1
V = np.arange(vlb, vth, dV)

#as far as in know, these dont matter
tau_x = 200
Vx = -85
gx = 0
Dx = 8
Vxh = -40
xi = 1/(1+np.exp(-(V-Vxh)/Dx))

mu0 = params['mxe']
var = params['vxe']
D = gl*np.sqrt(2*var*Cm/gl)

taud = 0*np.ones((N,1)) #synaptic delay
tausyne = params['tausyne']
tausyni = params['tausyni']
taus = np.zeros((N,1))
taus[:N_e] = tausyne
taus[N_e:] = tausyni

Tmax = params['Tmax']
dt = params['dt']
u1 = 1
df = params['df']
fmax = params['fmax']
freq = params['freq']
# idx = np.argwhere(np.abs(freq) < 1e-6)
# freq[idx] = 1e-6
nfreq = len(freq)

xi = [x.item() for x in xi]
cell_prms = [gl,Cm,DeltaT,vT,vl,vth,vlb,dV,vre,tref,tau_x,Vx,gx]

#Generate the connectivity matrix
net = np.load(save_dir + 'mc_eif_ucpld_weights.npz')['arr_0']

##################################################
## Solve Fokker-Planck for the uncoupled case
##################################################

tup = _rsnn.fp_eif(cell_prms + [mu0,var,xi])
P0,p0,J0,x0,r0 = tup

# #Solve Fokker-Planck for GWN
# M = 100
# mu_arr = np.linspace(0,4,M)
# var = 81
# At_arr = np.zeros((M,nfreq),dtype=np.cdouble)
#
# for i, mu in enumerate(mu_arr):
#     tup = hebb_backend.fp_eif(cell_prms + [mu0,var,xi])
#     P0,p0,J0,x0,r0 = tup
#     params = cell_prms + [x0,mu,var,u1,r0,P0,xi,p0,freq,nfreq]
#     V1r,V1i,x1r,x1i,Ar,Ai,alpha,beta = hebb_backend.lr_eif(params)
#     At = np.array(Ar) + np.array(Ai)*1j
#     At_arr[i,:] = At

########################
# Write objects to disk
########################

np.savez_compressed(save_dir + 'lr_eif_ucpld_P0', P0)
np.savez_compressed(save_dir + 'lr_eif_ucpld_r0', r0)
# np.savez_compressed(save_dir + 'lr_eif_ucpld_At', At_arr)
