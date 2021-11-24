import numpy as np
import matplotlib.pyplot as plt

save_dir = '/home/cwseitz/Desktop/data/'
npzfile = 'EIFLR.npz'
npzfile = np.load(save_dir + npzfile)

At = npzfile['arr_0']
Ft = npzfile['arr_1']
Ct0 = npzfile['arr_2']
Conn = npzfile['arr_3']

NE = 1000
NI = 0
N = NE + NI
d = 0
C = 1
var = 81
gL = 0.1
D = gL*np.sqrt(2*var*C/gL)
u1 = 1
nfreq = At.shape[-1]
Ct = np.zeros((N,N,nfreq),dtype=np.cdouble)
I = np.eye(N);

for j in range(nfreq):

    print(f'Computing C(w) for w = {j}')
    K = np.zeros((N,N),dtype=np.cdouble)
    yy0 = np.zeros((N,N))
    for k in range(N):
        for l in range(N):
            K[k,l] = Conn[k,l]*At[k,j]*Ft[l,j]
        yy0[k,k] = Ct0[k,j]
    Ct[:,:,j] = (I-K) @ yy0 @ (I-K.conj().T)
np.savez_compressed(save_dir + 'EIFC', Ct)
