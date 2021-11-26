import numpy as np
import matplotlib.pyplot as plt
import hebb_backend
from hebb.models import *

##################################################
## Predicting spike train cross-correlations as
## for excitatory-inhibitory random network
## using the linear response approximation
##################################################
## Author: Clayton Seitz
## Copyright: 2021, The Hebb Project
## Email: cwseitz@uchicago.edu
##################################################


# SA = 2.5*10^-4
NE = 1000
NI = 0
N = NE + NI
gL = 0.1
C = 1
Delta = 1.4
VT = -48
Vth = 30
Vlb = -100
dV = .1
Vr = -72
VL = Vr
tref = 2

tau_x = 200
Vx = -85
gx = 0

V = np.arange(Vlb, Vth, dV)

Dx = 8
Vxh = -40
xi = 1/(1+np.exp(-(V-Vxh)/Dx))

mu0 = 1
var = 81
D = gL*np.sqrt(2*var*C/gL)

taud = 0*np.ones((N,1)) #synaptic delays for output of each cell, ms
tauE = 5; tauI=2
taus = np.zeros((N,1))
taus[1:NE]=tauE; taus[NE+1:N]=tauI; #time constants for synaptic output of each cell, ms

Tmax = 50 #Maximum time lag over which to calculate cross-correlations (ms)
dt = 1 #Bin size for which to calculate cross-correlations (ms)
u1 = 1
df = 1/2/Tmax
fmax = 1/2/dt
freq = np.arange(-fmax, fmax, df)
idx = np.argwhere(np.abs(freq) < 1e-6)
freq[idx] = 1e-6
freq = freq.tolist()
nfreq = len(freq)

xi = [x.item() for x in xi]
params = [N,gL,C,Delta,VT,VL,Vth,Vlb,dV,Vr,tref,tau_x,Vx,gx,mu0,var,xi]

#Generate the connectivity matrix
adj = np.random.uniform(0,1,size=(N,N))
pref = 0.15*np.ones((N,N))
adj[adj > pref] = 0
adj[adj != 0] = 1
np.fill_diagonal(adj, 0)
p0 = np.sum(adj)/(N**2)
wmaxE = 5/(N*p0);
p_cond = wmaxE*.25;
adj[:NE,:NE]=p_cond*adj[:NE,:NE]

##################################################
## Fixed point iteration to find the rates
##################################################

tup = hebb_backend.fp_eif(params)
P0,p0,J0,x0,r0 = tup

num_rate_fp_its = 20
rates = r0*np.ones((N,))
for i in range(1,num_rate_fp_its):
    tmp_rates = []
    for j in range(N):
        mu = mu0 + np.sum(adj[j,:]*rates)
        params = [N,gL,C,Delta,VT,VL,Vth,Vlb,dV,Vr,tref,tau_x,Vx,gx,mu,var,xi]
        tup = hebb_backend.FokkerPlanck_EIF(params)
        tmp_rates.append(tup[4])
    rates = np.array(tmp_rates)

####################################################################
## Find linear response and synaptic kernel in the frequency domain
####################################################################

At = np.zeros((N,nfreq),dtype=np.cdouble)
Ft = np.zeros((N,nfreq),dtype=np.cdouble)
Ct0 = np.zeros((N,nfreq))

tup = hebb_backend.fp_eif(params)
P0,p0,J0,x0,r0 = tup

for i in range(1,N):

    print(f'Neuron {i}')
    mu_in = mu0 + np.sum(adj[i,:]*rates)
    params = [N,gL,C,Delta,VT,VL,Vth,Vlb,dV,Vr,tref,tau_x,Vx,gx,x0,mu_in,var,u1,r0,P0,xi,p0,freq,nfreq]
    V1r,V1i,x1r,x1i,Ar,Ai = hebb_backend.lr_eif(params)

    params = [N,gL,C,Delta,VT,VL,Vth,Vlb,dV,Vr,tref,tau_x,Vx,gx,x0,mu_in,var,u1,r0,freq,nfreq]

    f0r,f0i = hebb_backend.fpt_eif(params)
    f0 = np.array(f0r) + np.array(f0i)*1j
    C0 = r0*(1+2*np.real(f0/(1-f0)))

    Ft[i,:] = np.exp(1j*-2*np.pi*np.array(freq)*taud[i])/(1+1j*2*np.pi*np.array(freq)*taus[i])
    At[i,:] = np.array(Ar) + np.array(Ai)*1j
    Ct0[i,:] = C0

# save_dir = '/home/cwseitz/Desktop/data/'
# np.savez_compressed(save_dir + 'EIFLR', At, Ft, Ct0, adj)
