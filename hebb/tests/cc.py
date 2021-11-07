import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, ifft, fftfreq, fftshift
from scipy.signal import correlate, fftconvolve

N = 1000
y1 = np.random.normal(0,1,size=(N,)) #white noise
y2 = np.random.normal(0,1,size=(N,)) #white noise

#compute autocorrelation of the first signal
z1 = fft(y1)
fft1 = np.real(ifft(z1*z1.conj()))
fft1 = fftshift(fft1)

#fft1 = fftconvolve(y1,np.flip(y1), mode='same')
conv1 = correlate(y1,y1,method='direct',mode='same')

#compute autocorrelation of the second signal
z2 = fft(y2)
fft2 = np.real(ifft(z2*z2.conj()))
fft2 = fftshift(fft2)

#fft2 = fftconvolve(y2,np.flip(y2), mode='same')
conv2 = correlate(y2,y2,method='direct',mode='same')

#compute cross-correlation of the signals
fft3 = np.real(ifft(z1*z2.conj()))
fft3 = fftshift(fft3)

#fft3 = fftconvolve(y1,np.flip(y2), mode='same')
conv3 = correlate(y1,y2,method='direct',mode='same')

fig, ax = plt.subplots(1,2)
ax[0].plot(fft1)
ax[0].plot(fft2)
ax[0].plot(fft3)
ax[1].plot(conv1)
ax[1].plot(conv2)
ax[1].plot(conv3)
plt.show()
