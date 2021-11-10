import hebb_backend
import numpy as np
import matplotlib.pyplot as plt
import sys
np.set_printoptions(threshold=sys.maxsize)

q = 0.5
dt = 0.1
N = 20000
Nrecord = 100
T = 22000
Nt = int(round(T/dt))
N_e = int(round(q*N))
N_i = int(round((1-q)*N))
N_e1 = int(round(N_e/2))
N_i1 = int(round(N_i/2))


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

taux = 40
mxe = 0.015*np.sqrt(N)
mxi = 0.01*np.sqrt(N)
vxe = 0.05
vxi = 0.05

rxe = 0
rxi = 0
jeX = 0
jiX = 0
mxe0=mxe/np.sqrt(N)+rxe*jeX/N;
mxi0=mxi/np.sqrt(N)+rxi*jiX/N;

tausyne = 8.0
tausyni = 4.0
tausynx = 12.0

Jee=jee/np.sqrt(N)
Jei=-jei/np.sqrt(N)
Jie=jie/np.sqrt(N)
Jii=-jii/np.sqrt(N)

maxns = N*T*0.02

gl = [1/15, 1/10]
Cm = [1.0, 1.0]
vlb = [-100.0, -100.0]
vth = [-10.0, -10.0]
DeltaT = [2.0, 0.5]
vT = [-50.0, -50.0]
vl = [-60.0, -60.0]
vre = [-65.0, -65.0]
tref = [1.5, 0.5]

# V0min=vre[1]
# V0max=vT[1]
V0 = np.ones((N,))*5
V0 = list(V0)
V0 = [x.item() for x in V0]

sig = np.random.normal(0,1,size=(Nt,))
Ix1e = list(mxe+sig)
Ix1i = list(mxi+sig)
Ix2e = list(mxe+sig)
Ix2i = list(mxi+sig)

Irecord=list(np.sort(np.random.randint(0,N,Nrecord)))
Irecord = [x.item() for x in Irecord] #convert to native python type

c = [N,Nrecord,T,Nt,N_e,N_i,q,dt,pee0,pei0,pie0,pii0,jee,jei,jie,jii,
wee0,wei0,wie0,wii0,Kee,Kei,Kie,Kii,taux,mxe0,mxi0,vxe,vxi,tausyne,tausyni,
tausynx,Jee,Jei,Jie,Jii,maxns,gl,Cm,vlb,vth,DeltaT,vT,vl,vre,tref,Ix1e,
Ix2e,Ix1i,Ix2i,Nrecord,Irecord,V0,rxe,rxi,jeX,jiX,N_e1,N_i1]

vr, alphaer, alphair, alphaxr = hebb_backend.lif(c)
plt.plot(np.mean(vr,axis=0))
plt.show()
plt.plot(np.mean(alphaer,axis=0))
plt.plot(np.mean(alphair,axis=0))
plt.plot(np.mean(alphaxr,axis=0))
plt.show()
