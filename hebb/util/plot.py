import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib import cm
from operator import itemgetter
from .math import *

def plt2array(fig):
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    rgb_array_rgb = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3)
    return rgb_array_rgb

def add_k_ij_map(ax, M, sigma, delta=1):

    """
    Show the connectivity kernel of a neuron i projecting to all other neurons j
    """

    x0 = y0 = M/2
    im = np.zeros((M,M))
    xv, yv = np.meshgrid(np.arange(M),np.arange(M))
    X, Y = xv.ravel(), yv.ravel()
    for i in range(X.shape[0]):
        dx = torus_dist((x0,y0),(X[i],Y[i]),M,delta)
        im[X[i],Y[i]] = delta_gauss(dx, sigma, delta)
    ax.imshow(im, cmap='coolwarm')


def add_raster(ax, spikes, focal=None, trial=0, n_units=50):

    """
    Generate a raster plot by randomly selecting 'n_units'
    neurons from the tensor 'spikes'

    This function does not work well when a small number of units
    of a large population are spiking

    """

    units = np.random.choice(spikes.shape[0], n_units, replace=False)
    sub = spikes[units,:,:]
    arr = []
    for unit in units:
        spike_times = np.argwhere(spikes[unit,trial,:] > 0)
        spike_times = spike_times.reshape((spike_times.shape[0],))
        arr.append(spike_times)

    if focal is None:
        focal = n_units + 1
    colors = ['black' if i != focal else 'red' for i in range(n_units)]

    ax.eventplot(arr, colors='black', lineoffsets=1, linelengths=1)

def add_activity(ax, spikes, trial=0, color='red'):

    ax.plot(np.sum(spikes[:,trial,:], axis=0), color=color)

def add_ego_graph(ax, net, alpha=0.5):

    G = nx.convert_matrix.from_numpy_array(net.C, create_using=nx.DiGraph)
    node_and_degree = G.degree()
    (hub, degree) = sorted(node_and_degree, key=itemgetter(1))[-1]
    inedges = G.in_edges(hub)
    outedges = G.out_edges(hub)
    G = nx.Graph()
    G.add_node(hub)
    for neighbor in inedges:
        G.add_node(neighbor[0])
        G.add_edge(*neighbor, color='salmon')
    for neighbor in outedges:
        G.add_node(neighbor[1])
        G.add_edge(*neighbor, color='cornflowerblue')
    pos = nx.spring_layout(G)
    edges = G.edges()
    colors = [G[u][v]['color'] for u,v in edges]
    nx.draw(G, pos, ax=ax, alpha=alpha, node_color='black', edge_color=colors, node_size=20, with_labels=False)

def add_spectral_graph(ax, net, alpha=0.05, arrows=False):

    if arrows: arrows = True
    G = nx.convert_matrix.from_numpy_array(net.C, create_using=nx.DiGraph)
    pos = nx.spectral_layout(G)
    colors = []
    for n in G.nodes():
        if n in net.ex_idx:
            colors.append('red')
        else:
            colors.append('cornflowerblue')

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=20, node_shape='x')
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='black', alpha=alpha, arrows=arrows, arrowsize=10)

def add_spring_graph(ax, net, alpha=0.05, arrows=False):

    if arrows: arrows = True
    G = nx.convert_matrix.from_numpy_array(net.C, create_using=nx.DiGraph)
    pos = nx.spring_layout(G)
    colors = []
    for n in G.nodes():
        try:
            if n in net.ex_idx:
                colors.append('red')
            else:
                colors.append('cornflowerblue')
        except:
            colors.append('red')


    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=20, node_shape='x')
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='black', alpha=alpha, arrows=arrows, arrowsize=10)

def add_unit_voltage(ax, cell, unit=0, trial=0):

    ax.plot(cell.I[unit,trial,:], 'k')
    ax.grid(which='both')
    ax.set_ylabel('$\mathbf{PSP} \; [\mathrm{mV}]$')

def add_unit_current(ax, cell, unit=0, trial=0):

    ax.plot(cell.V[unit,trial,:], 'k')
    xmin, xmax = 0, cell.nsteps
    ax.hlines(cell.thr, xmin, xmax, color='red')
    ax.hlines(0, xmin, xmax, color='blue')
    ax.grid(which='both')
    ax.set_ylabel('$\mathbf{V}\; [\mathrm{mV}]$')

def add_unit_spikes(ax, cell, unit=0, trial=0):

    ax.plot(cell.Z[unit,trial,:], 'k')
    ax.grid(which='both')
    ax.set_ylabel('$\mathbf{Z}(t)$')

def add_unit_refrac(ax, cell, unit=0, trial=0):

    ax.plot(cell.R[unit,trial,cell.ref_steps:], 'k')
    ax.grid(which='both')
    ax.set_xlabel('t (ms)')
    ax.set_ylabel('$\mathbf{R}(t)$')

# def unit_i_stats(cell, unit=0, di=0.02):
#
#     """
#     Compute the histogram of current values for a single neuron over
#     trials, as a function of time i.e. P(I,t)
#     The vector over which P is calculated has shape (1, trials, 1)
#     """
#
#     fig, ax = plt.subplots(1,2)
#     for trial in range(cell.trials):
#         ax[0].plot(cell.I[unit,trial,:], color='black', alpha=0.1)
#     ax[0].set_ylabel('$\mathbf{PSP} \; [\mathrm{mV}]$')
#
#     bins = np.arange(0, 0.2, di)
#     colors = cm.coolwarm(np.linspace(0,1,cell.nsteps))
#     for t in range(cell.nsteps):
#         vals, bins = np.histogram(cell.I[unit,:,t], bins=bins)
#         vals = vals/(np.sum(vals)*di)
#         ax[1].plot(bins[:-1], vals, color=colors[t])
#
# def unit_v_stats(cell, dv=0.01):
#
#     """
#     Compute the histogram of voltage values for a single neuron over
#     trials, as a function of time i.e. P(V,t)
#     The vector over which P is calculated has shape (1, trials, 1)
#     """
#
#     bins = np.arange(0, cell.thr, dv)
#     temp = np.zeros((cell.nsteps,480,640,3))
#     imsave('data/temp.tif', temp)
#     im = pims.open('data/temp.tif')
#
#     h = np.apply_along_axis(lambda a: np.histogram(a, bins=bins, density=True)[0], 1, cell.V)
#     for t in range(cell.nsteps):
#         fig, ax = plt.subplots()
#         ax.imshow(h[:,:,t], cmap='coolwarm')
#         rgb_array_3d = plt2array(fig)
#         im[t] = rgb_array_3d

# def add_rate_hist(ax, cell, bins=20):
#
#     rates = np.mean(cell.Z,axis=1)
#     fig, ax = plt.subplots()
#     bins = np.linspace(rates.min(), rates.max(), bins)
#     colors = cm.coolwarm(np.linspace(0,1,cell.nsteps))
#     for t in range(cell.nsteps):
#         #idx = np.nonzero(clamp[:,0,t])
#         vals, bins = np.histogram(rates[:,t], bins=bins)
#         ax.plot(bins[:-1], vals, color=colors[t])
#
# def add_v_stats(ax, cell, dv=0.05):
#
#     """
#     Compute the histogram of voltage values over a population
#     as a function of time i.e. P(V,t)
#     """
#
#     bins = np.arange(0, cell.thr, dv)
#     fig, ax = plt.subplots()
#     colors = cm.coolwarm(np.linspace(0,1,cell.nsteps))
#     for t in range(cell.nsteps):
#         #idx = np.nonzero(cell.clamp[:,0,t])
#         vals, bins = np.histogram(cell.V[:,:,t], bins=bins)
#         vals = vals/(np.sum(vals)*dv)
#         ax.plot(bins[:-1], vals, color=colors[t])
