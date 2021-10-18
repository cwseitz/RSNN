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


def fig_3():

    def H(p_ij, p_ji):
        return p_ij*np.log(1/p_ij) + p_ji*np.log(1/p_ji) +\
              (1-p_ij)*(1-p_ji)*np.log(1/((1-p_ij)*(1-p_ji)))

    p_ijs = np.arange(0.1, 1, 0.01)
    p_jis = np.arange(0.1, 1, 0.01)
    mat = np.zeros((p_ijs.shape[0], p_ijs.shape[0]))

    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    X, Y = np.meshgrid(p_ijs, p_jis)
    zs = np.array(H(np.ravel(X), np.ravel(Y)))
    Z = zs.reshape(X.shape)

    ax1.plot_surface(X,Y,Z, cmap='coolwarm')
    ax1.view_init(azim=0, elev=90)
    ax1.set_xticks([0,1])
    ax1.set_xlabel(r'$p_{ij}$')
    ax1.set_ylabel(r'$p_{ji}$')
    ax1.set_yticks([0,1])
    ax1.set_zticks([])
    ax2.plot_surface(X,Y,Z, cmap='coolwarm')
    ax2.set_xlabel(r'$p_{ij}$')
    ax2.set_ylabel(r'$p_{ji}$')
    ax2.set_zlabel(r'$H(P)$')
    ax2.set_xticks([0,1])
    ax2.set_yticks([0,1])
    ax2.set_zticks([0,1])
    plt.tight_layout()

def fig_4():

    #Generate several homogeneous Gaussian networks to examine statistics
    N = 525
    sigmas = np.array([1, np.sqrt(N)/4, np.sqrt(N)/2])
    delta = 1

    fig, ax = plt.subplots(1,3, figsize=(8,2.25))

    for i in range(sigmas.shape[0]):
        rho_max = sigmas[i]*np.sqrt(2*np.pi)*np.exp(delta**2/(2*sigmas[i]**2))
        rhos = np.linspace(1, rho_max, 100)
        avg_arr = []
        var_arr = []
        for j in range(rhos.shape[0]):
            avg_grid = hogn_avg_out_deg(N, sigmas[i], rhos[j], delta, x0=0, y0=0)
            var_grid = hogn_var_out_deg(N, sigmas[i], rhos[j], delta, x0=0, y0=0)
            avg_arr.append(np.sum(avg_grid))
            var_arr.append(np.sum(var_grid))
        ax[i].plot(rhos, avg_arr, color='red')
        ax2 = ax[i].twinx()
        ax2.plot(rhos, var_arr, color='blue')
        ax2.set_ylabel(r'$\mathrm{Var}(N_{ij})$', color='blue')

    ax[0].set_title(r'$\sigma=1$')
    ax[0].set_xlabel(r'$\rho$')
    ax[0].set_ylabel(r'$\langle N_{ij} \rangle$', color='red')
    ax[1].set_title(r'$\sigma=N^{2}/4$')
    ax[1].set_xlabel(r'$\rho$')
    ax[1].set_ylabel(r'$\langle N_{ij} \rangle$', color='red')
    ax[2].set_title(r'$\sigma=N^{2}/2$')
    ax[2].set_ylabel(r'$\langle N_{ij} \rangle$', color='red')
    ax[2].set_xlabel(r'$\rho$')
    plt.tight_layout()
    plt.show()
