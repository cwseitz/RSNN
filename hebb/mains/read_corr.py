import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob

save_dir = '/home/cwseitz/Desktop/data/'
npz_file = np.load(save_dir + 'EIFC.npz')
Ct = npz_file['arr_0']
Ct_real = np.real(Ct)
Ct_imag = np.imag(Ct)
Ct_abs = np.abs(Ct)

#Transform to time domain
C = np.fft.ifft(Ct,axis=-1)

# avg_r = np.mean(Ct_real,axis=(0,1))
# avg_i = np.mean(Ct_imag,axis=(0,1))
# avg_mag = np.mean(Ct_abs,axis=(0,1))
avg = np.mean(C, axis=(0,1))
avg = np.roll(avg, avg.shape[0]//2)
plt.plot(avg)
plt.show()
# plt.plot(avg_r)
# plt.plot(avg_i)
# plt.plot(avg_mag)
# plt.show()
