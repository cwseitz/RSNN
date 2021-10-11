import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft
from scipy.signal import find_peaks

def simple_fft(freq=20,
               fft_size=2048,
               nsamples=50,
               period=1):

    sample_rate = nsamples/period
    tres = 1/sample_rate #time resolution
    fres = sample_rate/fft_size #frequency resolution

    # \\\
    # ~~~~~~Generate signal, do FFT~~~~~
    # \\\

    t = np.linspace(0, period, nsamples)
    x = np.sin(2*np.pi*freq*t)
    f = np.arange(fft_size)*fres
    mag = np.abs(fft(x,n=fft_size))
    peaks, props = find_peaks(mag, height=.5*mag.max())
    peaks = fres*peaks

    # \\\
    # ~~~~~~Plot results~~~~~
    # \\\

    fig, ax = plt.subplots(1, 2, figsize=(12,5))
    ax[0].plot(t, x)
    ax[0].set_xlabel('Time', fontsize=14)
    ax[0].set_ylabel('Amplitude', fontsize=14)
    ax[1].plot(f, mag)
    ax[1].set_xlabel('Frequency (Hz)', fontsize=14)
    ax[1].set_ylabel('a.u.', fontsize=14)
    plt.show()

    # \\\
    # ~~~~~~Print results~~~~~
    # \\\

    tres = tres*1e4
    mag[fft_size//2:] = 0

    print('#'*50)
    print('Time Resolution: %s ms/step' % tres)
    print('Frequency Resolution: %s Hz/step' % fres)
    print('Set Frequency: %s Hz' % freq)
    print('Measured Frequency: %s Hz' % peaks[0])


sample_rate = 1000 #Hz
period = .1 #sec
nsamples = int(round(sample_rate*period))

simple_fft(freq=100,
           fft_size=2048,
           nsamples=nsamples,
           period=period)
