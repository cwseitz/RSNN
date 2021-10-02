import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

def plt2array(fig):
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    rgb_array_rgb = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3)
    return rgb_array_rgb

def plot_activity(cell, trial=0):

    fig, ax = plt.subplots(2,1, sharex=True)
    ax[0].imshow(cell.V[:,trial,:], cmap='gray')
    ax[0].set_ylabel('N')
    ax[1].imshow(np.mod(cell.Z[:,trial,:]+1,2), cmap='gray')
    ax[1].set_ylabel('N')
    plt.legend()

def plot_rate_hist(cell, bins=20):

    rates = np.mean(cell.Z,axis=1)
    fig, ax = plt.subplots()
    bins = np.linspace(rates.min(), rates.max(), bins)
    colors = cm.coolwarm(np.linspace(0,1,cell.nsteps))
    for t in range(cell.nsteps):
        #idx = np.nonzero(clamp[:,0,t])
        vals, bins = np.histogram(rates[:,t], bins=bins)
        ax.plot(bins[:-1], vals, color=colors[t])

def pop_v_stats(cell, dv=0.05):

    """
    Compute the histogram of voltage values over a population
    as a function of time i.e. P(V,t)
    """

    bins = np.arange(0, cell.thr, dv)
    fig, ax = plt.subplots()
    colors = cm.coolwarm(np.linspace(0,1,cell.nsteps))
    for t in range(cell.nsteps):
        #idx = np.nonzero(cell.clamp[:,0,t])
        vals, bins = np.histogram(cell.V[:,:,t], bins=bins)
        vals = vals/(np.sum(vals)*dv)
        ax.plot(bins[:-1], vals, color=colors[t])

def unit_i_stats(cell, unit=0, di=0.02):

    """
    Compute the histogram of current values for a single neuron over
    trials, as a function of time i.e. P(I,t)
    The vector over which P is calculated has shape (1, trials, 1)
    """

    fig, ax = plt.subplots(1,2)
    for trial in range(cell.trials):
        ax[0].plot(cell.I[unit,trial,:], color='black', alpha=0.1)
    ax[0].set_ylabel('$\mathbf{PSP} \; [\mathrm{mV}]$')

    bins = np.arange(0, 0.2, di)
    colors = cm.coolwarm(np.linspace(0,1,cell.nsteps))
    for t in range(cell.nsteps):
        vals, bins = np.histogram(cell.I[unit,:,t], bins=bins)
        vals = vals/(np.sum(vals)*di)
        ax[1].plot(bins[:-1], vals, color=colors[t])

def unit_v_stats(cell, dv=0.01):

    """
    Compute the histogram of voltage values for a single neuron over
    trials, as a function of time i.e. P(V,t)
    The vector over which P is calculated has shape (1, trials, 1)
    """

    bins = np.arange(0, cell.thr, dv)
    temp = np.zeros((cell.nsteps,480,640,3))
    imsave('data/temp.tif', temp)
    im = pims.open('data/temp.tif')

    h = np.apply_along_axis(lambda a: np.histogram(a, bins=bins, density=True)[0], 1, cell.V)
    for t in range(cell.nsteps):
        fig, ax = plt.subplots()
        ax.imshow(h[:,:,t], cmap='coolwarm')
        rgb_array_3d = plt2array(fig)
        im[t] = rgb_array_3d

def plot_unit(cell, unit=0, trial=0):

    #Plot input and state variables for a single unit in a single trial
    fig, ax = plt.subplots(4,1, sharex=True)

    ax[0].plot(cell.I[unit,trial,:], 'k')
    xmin, xmax = ax[0].get_xlim()
    ax[0].grid(which='both')
    ax[0].set_ylabel('$\mathbf{PSP} \; [\mathrm{mV}]$')

    ax[1].plot(cell.V[unit,trial,:], 'k')
    ax[1].hlines(cell.thr, xmin, xmax, color='red')
    ax[1].hlines(0, xmin, xmax, color='blue')
    ax[1].grid(which='both')
    ax[1].set_ylabel('$\mathbf{V}\; [\mathrm{mV}]$')

    ax[2].plot(cell.Z[unit,trial,:], 'k')
    ax[2].grid(which='both')
    ax[2].set_ylabel('$\mathbf{Z}(t)$')

    ax[3].plot(cell.R[unit,trial,cell.ref_steps:], 'k')
    ax[3].grid(which='both')
    ax[3].set_xlabel('t (ms)')
    ax[3].set_ylabel('$\mathbf{R}(t)$')
    plt.tight_layout()
