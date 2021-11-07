import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, ifft, fftfreq, fftshift
from scipy.signal import correlate, fftconvolve

def cc(y1,y2):

    xv, yv = np.meshgrid(np.arange(0,y1.shape[0],1),np.arange(0,y1.shape[0],1))
    X,Y = xv.ravel(), yv.ravel()
    z_arr = []
    for i in range(y1.shape[1]):
        #trials
        z_sub_arr = []
        for j in range(X.shape[0]):
            z1 = fft(y1[X[j],i,:],axis=-1)
            z2 = fft(y2[Y[j],i,:],axis=-1)
            z3 = np.real(ifft(z1*z2.conj()))
            z_sub_arr.append(z3)
        z_arr.append(np.array(z_sub_arr))
    z_arr = np.array(z_arr)
    z_arr = np.swapaxes(z_arr,0,1)
    return z_arr

def cc_vec(y1,y2):

    z1 = fft(y1,axis=-1)
    z2 = fft(y2,axis=-1)
    z_arr = []

    #iterate over trials
    for i in range(y1.shape[1]):
        x1 = z1[:,i,:]
        x2 = z2[:,i,:].conj()
        x3 = np.einsum('ij,kj->ikj',x1,x2)
        x3 = x3.reshape((x2.shape[0]*x2.shape[0],x2.shape[1]))
        z3 = np.real(ifft(x3))
        z_arr.append(z3)
    z_arr = np.array(z_arr)
    z_arr = np.swapaxes(z_arr,0,1)
    return z_arr

N = 100; batches = 50; nsteps = 1000
y1 = np.random.normal(0,1,size=(N,batches,nsteps)) #white noise

z1 = cc_vec(y1,y1)
# z2 = cc(y1,y1)
z1_mean = np.mean(z1, axis=(0,1))
z2_mean = np.mean(z2, axis=(0,1))
plt.plot(z1_mean)
plt.plot(z2_mean)
plt.show()
