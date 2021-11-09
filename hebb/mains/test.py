import hebb_backend
import numpy as np

q = 0.5
dt = 0.001
N = 20000
Nrecord = 100
T = 1
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

tausyne = 8
tausyni = 4
tausynx = 12

Jee=jee/np.sqrt(N)
Jei=-jei/np.sqrt(N)
Jie=jie/np.sqrt(N)
Jii=-jii/np.sqrt(N)

maxns = N*T*0.02
gl = [1/15, 1/10]
Cm = [1, 1]
vlb = [-100, -100]
vth = [-10, -10]
DeltaT = [2, .5]
vT = [-50, -50]
vl = [-60, -60]
vre = [-65, -65]
tref = [1.5, .5]

# Kx = np.sqrt((vxe/taux)*np.sqrt(2/np.pi))*exp(-(-6*taux:dt:6*taux).^2./((taux)^2));
# sig = dt*conv(randn(Nt,1)./sqrt(dt),Kx,'same')
sig = np.random.normal(0,1,size=(Nt,))
Ix1e = list(mxe+sig)
Ix1i = list(mxi+sig)
Ix2e = list(mxe+sig)
Ix2i = list(mxi+sig)

nrecord=500;
Irecord=list(np.sort(np.random.randint(0,N_e,nrecord)))

c = [
N,
Nrecord,
T,
Nt,
N_e,
N_i,
q,
dt,
pee0,
pei0,
pie0,
pii0,
jee,
jei,
jie,
jii,
wee0,
wei0,
wie0,
wii0,
Kee,
Kei,
Kie,
Kii,
taux,
mxe0,
mxi0,
vxe,
vxi,
tausyne,
tausyni,
tausynx,
Jee,
Jei,
Jie,
Jii,
maxns,
gl,
Cm,
vlb,
vth,
DeltaT,
vT ,
vl,
vre,
tref,
Ix1e,
Ix2e,
Ix1i,
Ix2i,
nrecord,
Irecord
]

# Ix1e,Ix2e,Ix1i,Ix2i,Ne,Ni,Ne1,Ni1,Jex,Jix,Jee,Jei,Jie,Jii,
#              rxe,rxi,Kee,Kei,Kie,Kii,Cm,gl,vl,DeltaT,vT,tref,vth,vre,vlb,
#              tausynx,tausyne,tausyni,V0,T,dt,maxns,Irecord

d = hebb_backend.lif(c)
