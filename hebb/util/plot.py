import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from .util import *


def plot_input_statistics(v, bins=10):

    """

    Plot the distribution of voltage updates at each time step

    Parameters
    ----------
    v : ndarray
        A tensor, where each element is the membrane potential of a neuron
        at a particular time. Has shape (unit, batch, time)

    Returns
    -------
    None

    """

    fig, ax = plt.subplots()

    nsteps = v.shape[-1]
    colormap = cm.get_cmap('coolwarm')
    colors = colormap(np.linspace(0, 1, nsteps))
    #compute the first difference in the voltage along time axis
    v_diff = np.diff(v, axis=-1)
    #compute the histogram of values over (unit, batch) matrix
    hist_arr, edges_arr = [], []
    for t in range(nsteps):
        hist, edges = np.histogram(v_diff[:,:,t-1], bins=bins, density=True)
        plt.plot(edges[:-1], hist, color=colors[t], alpha=0.5)
    ax.set_title('Input Statistics')
    ax.set_xlabel('Voltage (a.u.)')
    ax.set_ylabel('PDF')
    plt.tight_layout()

def plot_voltage_statistics(v, bins=10):

    """

    Plot the distribution of voltages at each time step

    Parameters
    ----------
    v : ndarray
        A tensor, where each element is the membrane potential of a neuron
        at a particular time. Has shape (unit, batch, time)

    Returns
    -------
    None

    """

    fig, ax = plt.subplots()

    units, batches, nsteps = v.shape
    colormap = cm.get_cmap('coolwarm')
    colors = colormap(np.linspace(0, 1, nsteps))

    #compute the histogram of values over (unit, batch) matrix
    hist_arr, edges_arr = [], []
    for t in range(nsteps):
        hist, edges = np.histogram(v[:,:,t], bins=bins, density=True)
        ax.plot(edges[:-1], hist, color=colors[t], alpha=0.5)

    ax.set_xlabel('Voltage (a.u.)')
    ax.set_ylabel('PDF')
    plt.tight_layout()


def plot_weights(net):

    """

    Plot the input weight matrix and recurrent weight matrix

    Parameters
    ----------
    net : object
        A LIF network object

    Returns
    -------
    None

    """

    fig, ax = plt.subplots(1,2)
    ax[0].imshow(net.in_weights, cmap='gray')
    ax[1].imshow(net.rec_weights, cmap='gray')
    plt.tight_layout()
