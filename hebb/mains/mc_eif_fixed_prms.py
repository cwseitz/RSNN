import json
import numpy as np
from hebb.models import *

##################################################
## Set params for a Monte Carlo simulation for
## excitatory-inhibitory fixed network
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
N = 10000
Nrecord = 100
T = 10000
Nt = int(round(T/dt))
N_e = int(round(q*N))
N_i = int(round((1-q)*N))

pee0 = 0.25
pei0 = 0.25
pie0 = 0.25
pii0 = 0.25

jee = 12.5
jei = 50
jie = 20
jii = 50

wee0 = jee*pee0*q
wei0 = jei*pei0*(1-q)
wie0 = jie*pie0*q
wii0 = jii*pii0*(1-q)

Kee = pee0*N_e
Kei = pei0*N_e
Kie = pie0*N_i
Kii = pii0*N_i

Jee=jee/np.sqrt(N)
Jei=-jei/np.sqrt(N)
Jie=jie/np.sqrt(N)
Jii=-jii/np.sqrt(N)

maxns = N*T*0.02

########################
## Define neuron params
########################

gl = [1/15, 1/10]
Cm = [1.0, 1.0]
vlb = [-100.0, -100.0]
vth = [-10.0, -10.0]
vl = [-60.0, -60.0]
DeltaT = [2.0, 0.5]
vT = [-50.0, -50.0]
vre = [-65.0, -65.0]
tref = [1.5, 0.5]
tausyne = 8.0
tausyni = 4.0
tausynx = 12.0

########################
## Define stim params
########################

taux = 40
mxe = 0.015*np.sqrt(N)
mxi = 0.01*np.sqrt(N)
vxe = 0.05
vxi = 0.05

mxe0=(mxe/np.sqrt(N))
mxi0=(mxi/np.sqrt(N))

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
'pee0':pee0,
'pii0':pii0,
'pei0':pei0,
'pie0':pie0,
'jee':jee,
'jii':jii,
'jei':jei,
'jie':jie,
'wee0':wee0,
'wii0':wii0,
'wei0':wei0,
'wie0':wie0,
'Kee':Kee,
'Kii':Kii,
'Kei':Kei,
'Kie':Kie,
'Jee':Jee,
'Jii':Jii,
'Jei':Jei,
'Jie':Jie,
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
'mxi0':mxi0

}

with open(save_dir + 'params.json', 'w') as fp:
    json.dump(params, fp)
