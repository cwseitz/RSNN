import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import cm
from hebb.util import *

def fig_1(net):

    custom_lines = [Line2D([0],[0],color='salmon', lw=4),Line2D([0],[0],color='cornflowerblue', lw=4)]

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax0 = ax.inset_axes([0, 0, 0.5, 1.0])
    ax1 = ax.inset_axes([0.6, 0.55, 0.2, 0.4])
    ax2 = ax.inset_axes([0.9, 0.55, 0.2, 0.4])
    ax3 = ax.inset_axes([0.6, 0, 0.2, 0.4])
    ax4 = ax.inset_axes([0.9, 0, 0.2, 0.4])

    add_spectral_graph(ax0, net)

    # ax1.plot(net.N[5,:], color='purple')
    # ax1.plot(net.N[25,:], color='blue')
    # ax1.plot(net.N[50,:], color='red')
    # ax1.plot(net.N[75,:], color='cyan')
    # ax1.set_xlabel(r'$\mathbf{\Delta}_{ij}$')
    # ax1.set_ylabel(r'$\langle\mathbf{N_{ij}}\rangle$')
    #
    # ax2.plot(net.N_var[5,:], color='purple')
    # ax2.plot(net.N_var[25,:], color='blue')
    # ax2.plot(net.N_var[50,:], color='red')
    # ax2.plot(net.N_var[75,:], color='cyan')
    # ax2.set_ylabel(r'$Var\;(\mathbf{N_{ij}})$')
    # ax2.set_xlabel(r'$\mathbf{\Delta}_{ij}$')
    plt.tight_layout()

def fig_2(lif, net, spikes, focal=0):

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax0 = ax.inset_axes([0, 0.65, 0.4, 0.45])
    ax1 = ax.inset_axes([0.5, 0.9, 0.5, 0.2])
    ax2 = ax.inset_axes([0.5, 0.65, 0.5, 0.2])
    ax3 = ax.inset_axes([0, 0.4, 1, 0.15])
    ax4 = ax.inset_axes([0, 0.2, 1, 0.15])
    ax5 = ax.inset_axes([0, 0, 1, 0.15])

    add_spectral_graph(ax0, net.CIJ, net.in_idx)
    add_raster(ax1, spikes, n_units=100)
    add_activity(ax2, spikes)
    add_unit_voltage(ax3, lif, unit=focal)
    add_unit_current(ax4, lif, unit=focal)
    add_unit_spikes(ax5, lif, unit=lif.no_clamp_idx[focal])
    plt.tight_layout()
