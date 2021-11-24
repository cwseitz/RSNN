import numpy as np
import matplotlib.pyplot as plt
import hebb_backend
from hebb.models import *

##################################################
## Predicting spike train cross-correlations as
## a function of synaptic connectivity
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

Tmax = 300 #Maximum time lag over which to calculate cross-correlations (ms)
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
p_xx = [0.25,0.25,0.25,0.25]
J_xx = [0.1, 0.1, 0.1, 0.1]
p_e = 0.5
net = ExInFixedNetwork(N, p_e, p_xx, J_xx)

##################################################
## Fixed point iteration to find the rates
##################################################

tup = hebb_backend.FokkerPlanck_EIF(params)
P0,p0,J0,x0,r0 = tup

num_rate_fp_its = 20
rates = r0*np.ones((N,))
for i in range(1,num_rate_fp_its):
    tmp_rates = []
    for j in range(N):
        mu = mu0 + np.sum(net.C[j,:]*rates)
        params = [N,gL,C,Delta,VT,VL,Vth,Vlb,dV,Vr,tref,tau_x,Vx,gx,mu,var,xi]
        tup = hebb_backend.FokkerPlanck_EIF(params)
        tmp_rates.append(tup[4])
    rates = np.array(tmp_rates)

####################################################################
## Find linear response and synaptic kernel in the frequency domain
####################################################################

At = np.zeros((N,nfreq))
Ft = np.zeros((N,nfreq))
Ct0 = np.zeros((N,nfreq))

tup = hebb_backend.FokkerPlanck_EIF(params)
P0,p0,J0,x0,r0 = tup

for i in range(1,N):

    mu_in = mu0 + np.sum(net.C[i,:]*rates)
    params = [N,gL,C,Delta,VT,VL,Vth,Vlb,dV,Vr,tref,tau_x,Vx,gx,x0,mu_in,var,u1,r0,P0,xi,p0,freq,nfreq]
    V1r,V1i,x1r,x1i,Ar,Ai = hebb_backend.LA_EIF(params)

    params = [N,gL,C,Delta,VT,VL,Vth,Vlb,dV,Vr,tref,tau_x,Vx,gx,x0,mu_in,var,u1,r0,freq,nfreq]
    f0r,f0i = hebb_backend.FPT_EIF(params)
    f0r,f0i = np.array(f0r),np.array(f0i)
    C0 = r0*(1+2.*(f0r/(1-f0r)))

    Ft[i,:] = np.exp(1j*-2*np.pi*np.array(freq)*taud[i])/(1+1j*2*np.pi*np.array(freq)*taus[i])
    At[i,:] = np.array(Ar) + np.array(Ai)*1j
    Ct0[i,:] = C0

# def eif_sim(N,Nt,gL,C,Delta,VT,Vth,dV,Vr,VL,mu,var):
#
#     V = np.zeros((N,Nt))
#     V[:,0] = Vr
#     D = np.sqrt(2*C/gL)
#     for i in range(N):
#         for j in range(Nt):
#             x = gL*np.sqrt(var)*D*np.random.normal(0,1) + mu
#             V[i,j] = V[i,j-1] + (gL*(VL-V[i,j-1]) + gL*Delta*np.exp((V[i,j-1]-VT)/Delta) + x)/C
#             if V[i,j] > Vth:
#                 V[i,j] = Vr
#
#     return V
#
# N = 1000
# Nt = 1000
# mu = mu_vec[0]
# var = sig2_vec[0]
# V_mc = eif_sim(N,Nt,gL,C,Delta,VT,Vth,dV,Vr,VL,mu,var)
#
#
# plt.plot(V,P0,color='blue')
# plt.hist(V_mc[:,-1],density=True, bins=25, color='black', alpha=0.5)
# plt.show()
