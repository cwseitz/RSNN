import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from matplotlib import cm
from operator import itemgetter
from ..math import *

##################################################
## Library of functions that add specific subplots
## to an axis specified by the user
##################################################
## Author: Clayton Seitz
## Copyright: 2021, The Hebb Project
## Email: cwseitz@uchicago.edu
##################################################

"""
Neuron state variables in time
"""

def add_avg_current(ax, curr, dt, color='red'):

    Nt = curr.shape[-1]
    t = np.arange(0,Nt,1)*dt
    ax.plot(t, np.mean(curr,axis=0), color=color)

def add_unit_voltage(ax, v, dt, unit=0, trial=0, color='black'):

    """
    Add the voltage trace for a single neuron

    Parameters
    ----------
    ax : object,
        matplotlib axis object
    rnn : object,
        RNN object
    unit : int, optional
        index of unit to plot
    trial : int, optional
        index of trial to plot
    """

    Nt = v.shape[-1]
    t = np.arange(0,Nt,1)*dt
    ax.plot(t, v[unit,trial,:], color=color)
    ax.grid(which='both')

def add_raster(ax, spikes, dt, color='black'):

    """
    Generate a raster plot by randomly selecting 'n_units'
    neurons from the tensor 'spikes'.

    **Note : This function does not work well when a small number of units
    of a large population are spiking

    Parameters
    ----------
    ax : object,
        matplotlib axis object
    spikes : ndarray
        tensor of spikes
    focal : int, optional
        index of unit to highlight in red
    trial : int, optional
        index of trial to plot
    n_units : int, optional
        number of units to plot raster, defaults to 50
    """

    units = spikes.shape[0]
    arr = []
    for unit in range(units):
        spike_times = np.argwhere(spikes[unit,:] > 0)
        spike_times = spike_times.reshape((spike_times.shape[0],))
        arr.append(spike_times*dt)
        ax.eventplot(arr, colors=color, orientation='horizontal', lineoffsets=1, linelengths=1)

def add_activity(ax, spikes, trial=0, color='red'):

    """
    Plot the population activity (the sum over units at each time step)

    Parameters
    ----------
    ax : object,
        matplotlib axis object
    spikes : ndarray
        tensor of spikes
    trial : int, optional
        index of trial to plot
    color : str, optional
        color for activity plot, defaults to red
    """

    ax.plot(np.sum(spikes[:,trial,:], axis=0), color=color)

def add_volt_hist(ax, v, dv=2):

    """
    Plot the histogram of voltages at a single time point

    Parameters
    ----------
    """

    N = v.shape[0]
    bins = np.arange(v.min(), v.max(), dv)
    vals, bins = np.histogram(v, bins=bins)
    vals = vals/(np.sum(vals)*dv) #normalize by integral
    ax.plot(bins[:-1], vals, color='purple',alpha=0.75)

def add_rate_hist(ax, spikes, dt, dr=10):

    """
    Plot the histogram of firing rates for excitatory and inhibitory neurons

    Parameters
    ----------
    """

    N = spikes.shape[0]
    rates = 1000*np.sum(spikes,axis=0)/(N*dt)
    bins = np.arange(rates.min(), rates.max(), dr)
    vals, bins = np.histogram(rates, bins=bins)
    vals = vals/(np.sum(vals)*dr) #normalize by integral
    ax.plot(bins[:-1], vals, color='black',alpha=0.5)
    return np.mean(rates)

def add_ffwd_hist(ax, ffwd):

    """
    Plot the average cross spectrum

    Parameters
    ----------
    """

    bins = np.arange(np.array(ffwd.Ix1e).min(),np.array(ffwd.Ix1e).max(),0.025)
    vals, bins = np.histogram(ffwd.Ix1e, bins=bins)
    vals = vals/(np.sum(vals)*0.025) #normalize by integral
    ax.plot(bins[:-1], vals, color='red', linestyle='--', label='E')

    bins = np.arange(np.array(ffwd.Ix1i).min(),np.array(ffwd.Ix1i).max(),0.025)
    vals, bins = np.histogram(ffwd.Ix1i, bins=bins)
    vals = vals/(np.sum(vals)*0.025) #normalize by integral
    ax.plot(bins[:-1], vals, color='blue', linestyle='--', label='I')

def add_curr_hist(ax, curr, min=-5, max=3, label=None, color='red'):

    """
    Add a histogram of synaptic currents to the axis

    Parameters
    ----------
    """

    bins = np.arange(min,max,0.1)
    vals, bins = np.histogram(curr, bins=bins)
    vals = vals/(np.sum(vals)*0.1) #normalize by integral
    ax.plot(bins[:-1], vals, alpha=0.5, label=label, color=color)

    # vals, bins = np.histogram(i_i, bins=bins)
    # vals = vals/(np.sum(vals)*0.1) #normalize by integral
    # ax.plot(bins[:-1], vals, alpha=0.5, color='blue')


def add_cc_hist(ax, x, dt, color='red', rand_select=300):

    """
    Plot a histogram of cross correlations

    Parameters
    ----------
    """

    n = x.shape[0]
    T = x.shape[-1]
    if rand_select != None:
        g = np.arange(0,n,1)
        v = np.random.choice(g,size=(int(rand_select),),replace=False)
        x = x[v,:,:]

    #indices of off-diagonal elements
    s = block_cc(x)
    idx_x, idx_y = np.where(~np.eye(s.shape[0],dtype=bool))
    s = s[idx_x,idx_y,:,:]

    bins = np.arange(0,1,0.02)
    vals, bins = np.histogram(s, bins=bins)
    vals = vals/(np.sum(vals)*0.02) #normalize by integral
    ax.plot(bins[:-1], vals, color=color)

# def add_mean_cc_brute(ax, z, color='red'):
#
#     """
#     Plot the average cross-correlation using the brute force method
#
#     Parameters
#     ----------
#     """
#
#     N = z.shape[0]
#     xv,yv = np.meshgrid(np.arange(0,N,1),np.arange(0,N,1))
#     X,Y = xv.ravel(), yv.ravel()
#     arr = []
#     for i in range(X.shape[0]):
#         x = X[i]; y = Y[i]
#         if x != y:
#             z_x = (z[x] - np.mean(z[x]))/np.sqrt(0.05)
#             z_y = (z[y] - np.mean(z[y]))/np.sqrt(0.05)
#             # xt = np.fft.fft(z_x); yt = np.fft.fft(z_y)
#             # cc = np.abs(np.fft.ifft(xt*yt.conj()))
#             cc = np.correlate(z_x,z_y,mode='same')
#             arr.append(cc)
#     arr = np.array(arr)
#     ax.plot(np.mean(arr,axis=0)/1000,color=color)


def add_mean_cc(ax, x, dt, color='red', linestyle='-'):

    """
    Plot the average cross-correlation

    Parameters
    ----------
    """

    Nt = x.shape[-1]
    t = np.arange(-Nt//2,Nt//2,1)*dt

    std = x.std(axis=-1, keepdims=True)
    mu = x.mean(axis=-1, keepdims=True)
    x = (x - mu)/std

    #indices of off-diagonal elements
    s = block_cc(x)
    idx_x, idx_y = np.where(~np.eye(s.shape[0],dtype=bool))
    s = s[idx_x,idx_y,:,:]
    avg = np.mean(s, axis=(0,1))
    avg = np.roll(avg, avg.shape[0]//2)
    ax.plot(t, avg, color=color,alpha=1, linestyle=linestyle)

def add_mean_ac(ax, x, dt, color='red', linestyle='-'):

    """
    Plot the average auto-correlation

    Parameters
    ----------
    """

    Nt = x.shape[-1]
    t = np.arange(-Nt//2,Nt//2,1)*dt

    std = x.std(axis=-1, keepdims=True)
    mu = x.mean(axis=-1, keepdims=True)
    x = (x - mu)/std

    #indices of off-diagonal elements
    s = block_cc(x)
    idx_x, idx_y = np.where(np.eye(s.shape[0],dtype=bool))
    s = s[idx_x,idx_y,:,:]
    avg = np.mean(s, axis=(0,1))
    avg = np.roll(avg, avg.shape[0]//2)
    ax.plot(t, avg, color=color,alpha=1, linestyle=linestyle)


def add_mean_cs(ax, x, dt, color='red', linestyle='-'):

    """
    Plot the average cross spectrum

    Parameters
    ----------
    """

    freq = np.fft.fftfreq(x.shape[-1], d=dt)
    T = x.shape[-1]

    #indices of off-diagonal elements
    s = block_spectra(x, magnitude=True)
    idx_x, idx_y = np.where(~np.eye(s.shape[0],dtype=bool))
    s = s[idx_x,idx_y,:,:]
    avg = np.mean(s, axis=(0,1))
    ax.plot(freq[1:T//2],avg[1:T//2], color=color, linestyle=linestyle)
    ax.set_ylim([0,avg[1:T//2].max()])

def add_mean_as(ax, x, dt, color='red', linestyle='-'):

    """
    Plot the mean autospectrum (standard power spectrum of one signal)

    Parameters
    ----------
    """

    freq = np.fft.fftfreq(x.shape[-1], d=dt)
    T = x.shape[-1]

    #indices of diagonal elements
    s = block_spectra(x, magnitude=True)
    idx_x, idx_y = np.where(np.eye(s.shape[0],dtype=bool))
    s = s[idx_x,idx_y,:,:]
    avg = np.mean(s, axis=(0,1))
    ax.plot(freq[1:T//2],avg[1:T//2], color=color, linestyle=linestyle)
    ax.set_ylim([0,avg[1:T//2].max()])


"""
Visualizing connectivity
"""

def add_kernel_pair(ax1, ax2, N, sigma, q=0.8):

    """
    Draw a pair of connectivity kernels converted to probabilities and
    the product of their probabilities

    Parameters
    ----------
    ax : object,
        matplotlib axis object
    net : object,
        network object
    alpha : float, optional
        transparency param
    """

    x1, y1 = np.sqrt(N)/2, np.sqrt(N)/4
    x2, y2 = np.sqrt(N)/2, 3*np.sqrt(N)/4
    k_ij = torgauss(N, x1, y1, sigma, delta=1)
    k_ji = torgauss(N, x2, y2, sigma, delta=1)
    p_ij, p_ji, p_x = trinomial(k_ij,k_ji,q)
    ax1.imshow(p_ij+p_ji, cmap='coolwarm')
    ax2.imshow(p_ij*p_ji, cmap='coolwarm')

def add_ego_graph(ax, net, alpha=0.5):

    """
    Draw an ego graph by selecting the node with the largest degree and
    drawing it and all of its neighbors

    Parameters
    ----------
    ax : object,
        matplotlib axis object
    net : object,
        network object
    alpha : float, optional
        transparency param
    """

    G = nx.convert_matrix.from_numpy_array(net.C, create_using=nx.DiGraph)
    node_and_degree = G.degree()
    (hub, degree) = sorted(node_and_degree, key=itemgetter(1))[-1]
    inedges = G.in_edges(hub)
    outedges = G.out_edges(hub)
    G = nx.Graph()
    G.add_node(hub)
    for neighbor in inedges:
        G.add_node(neighbor[0])
        G.add_edge(*neighbor, color='red')
    for neighbor in outedges:
        G.add_node(neighbor[1])
        G.add_edge(*neighbor, color='dodgerblue')
    pos = nx.spring_layout(G)
    edges = G.edges()
    colors = [G[u][v]['color'] for u,v in edges]
    nx.draw(G, pos, ax=ax, alpha=alpha, node_color='black', edge_color=colors, node_size=20, with_labels=False)

def add_spectral_graph(ax, net, alpha=0.05, sparse=False, arrows=False):

    """
    Draw a graph in spectral format

    Parameters
    ----------
    ax : object,
        matplotlib axis object
    net : object,
        network object
    alpha : float, optional
        transparency param
    arrows : bool, optional
        whether or not to draw the direction of an edge via arrows
    """


    if arrows:
        arrows = True
    if sparse:
        G = nx.convert_matrix.from_scipy_sparse_matrix(net.C, create_using=nx.DiGraph)
    else:
        G = nx.convert_matrix.from_numpy_array(net.C, create_using=nx.DiGraph)

    pos = nx.spectral_layout(G)
    colors = []
    for n in G.nodes():
        colors.append('dodgerblue')
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=20, node_shape='x')
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='black', alpha=alpha, arrows=arrows, arrowsize=10)

def add_spring_graph(ax, net, alpha=0.05, sparse=False, arrows=False):

    """
    Draw a graph in spring format

    Parameters
    ----------
    ax : object,
        matplotlib axis object
    net : object,
        network object
    alpha : float, optional
        transparency param
    arrows : bool, optional
        whether or not to draw the direction of an edge via arrows
    """

    if arrows:
        arrows = True
    if sparse:
        G = nx.convert_matrix.from_scipy_sparse_matrix(net.C, create_using=nx.DiGraph)
    else:
        G = nx.convert_matrix.from_numpy_array(net.C, create_using=nx.DiGraph)

    pos = nx.spring_layout(G)
    colors = []
    for n in G.nodes():
        colors.append('dodgerblue')
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=20, node_shape='x')
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='black', alpha=alpha, arrows=arrows, arrowsize=10)

def add_fractal_graph(ax, net, alpha=0.05):

    """
    Draw a fractal graph

    Parameters
    ----------
    ax : object,
        matplotlib axis object
    net : object,
        network object
    alpha : float, optional
        transparency param
    arrows : bool, optional
        whether or not to draw the direction of an edge via arrows
    """

    def level_mat(mx_lvl, sz_cl):
        level_mat = np.zeros((2**mx_lvl,2**mx_lvl), dtype=np.int8)
        i = 0
        for k in range(sz_cl+1,mx_lvl+1):
            i += 1
            for n in range(2**(mx_lvl-k)):
                level_mat[n*2**k:, :n*2**k] = i
                level_mat[:n*2**k, n*2**k:] = i
        return level_mat

    G = nx.convert_matrix.from_numpy_array(net.C)
    idxs = np.argwhere(net.C > 0)
    level_mat = level_mat(net.mx_lvl, net.sz_cl)

    colors = cm.coolwarm(np.linspace(0,1,net.mx_lvl-net.sz_cl))
    for idx in idxs:
        x,y = idx; color_idx = int(level_mat[x,y])
        G.edges[x,y]['color'] = colors[color_idx]

    colors = [G[u][v]['color'] for u,v in G.edges()]
    pos = nx.spring_layout(G)
    nx.draw(G, pos, ax=ax, alpha=alpha, node_size=5, node_color='black', edge_color=colors)
    plt.tight_layout()

"""
Visualizing statistical models
"""

def add_ou_hist(ax, ou, steps):

    ax.plot(ou._x, ou.p1[:,steps[0]], color='red', linestyle='--',)
    ax.plot(ou._x, ou.p1[:,steps[1]], color='blue', linestyle='--')
    ax.plot(ou._x, ou.p1[:,steps[2]], color='cyan', label='FP - 200ms', linestyle='--')

    ax.plot(ou._x, ou.p2[:,steps[0]], color='red')
    ax.plot(ou._x, ou.p2[:,steps[1]], color='blue')
    ax.plot(ou._x, ou.p2[:,steps[2]], color='cyan', label='Sim - 200ms')

    ax.set_xlim([-1, 1])
    ax.set_xlabel('X', fontsize=14)
    ax.set_ylabel('P(X)', fontsize=14)
    plt.tight_layout()
    plt.legend()
    plt.grid()
