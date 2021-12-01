import numpy as np
import matplotlib.pyplot as plt
import json

##################################################
##  Solve matrix equations for cross spectra
##################################################
## Author: Clayton Seitz
## Copyright: 2021, The Hebb Project
## Email: cwseitz@uchicago.edu
##################################################

save_dir = '/home/cwseitz/Desktop/data/'

with open(save_dir + 'params.json', 'r') as fh:
    params = json.load(fh)

lr_file = 'lr_eif_rand_lr.npz'
kern_file = 'lr_eif_rand_kern.npz'
c0_file = 'lr_eif_rand_c0.npz'
weight_file = 'mc_eif_rand_weights.npz'

At = np.load(save_dir + lr_file)['arr_0']
Ft = np.load(save_dir + kern_file)['arr_0']
Ct0 = np.load(save_dir + c0_file)['arr_0']
net = np.load(save_dir + weight_file)['arr_0']

N, Nfreq = At.shape
Ct = np.zeros((N,N,Nfreq),dtype=np.cdouble)
I = np.eye(N)

##################################################
## Solve for the cross spectra at each frequency
##################################################

for j in range(Nfreq):
    print(f'Computing C(w) for w = {j}')
    K = np.zeros((N,N),dtype=np.cdouble)
    yy0 = np.zeros((N,N))
    for k in range(N):
        for l in range(N):
            K[k,l] = net[k,l]*At[k,j]*Ft[l,j]
        yy0[k,k] = Ct0[k,j]
    Ct[:,:,j] = (I-K) @ yy0 @ (I-K.conj().T)
C = np.real(np.fft.ifft(Ct,axis=-1))

##################################################
## Save the predicted cross spectra to disk
##################################################

np.savez_compressed(save_dir + 'lr_eif_rand_ct', Ct)
np.savez_compressed(save_dir + 'lr_eif_rand_c', C)
