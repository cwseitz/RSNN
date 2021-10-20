import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import cm
from hebb.util import *
from hebb.models import *
from matplotlib.ticker import FormatStrFormatter

def fig_1():


    """
    Generate a homogeneous gaussian network and plot the graph in spectral format
    """

    N = 525
    net = HOGN(N, q=0.3)

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

    add_spring_graph(ax0, net)
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

    #Generate several homogeneous Gaussian networks to examine statistics
    N = 525
    M = np.sqrt(N)
    q = 0.2
    sigmas = np.array([1, M/2, np.sqrt(2)*M])
    delta = 1

    fig, ax = plt.subplots(1,3, figsize=(8,2.25))

    for i in range(sigmas.shape[0]):
        max_boost = sigmas[i]*np.sqrt(2*np.pi)*np.exp(delta**2/(2*sigmas[i]**2))
        boost = np.linspace(1, max_boost, 100)
        avg_arr = []; var_arr = []
        for j in range(boost.shape[0]):
            avg_grid = hogn_avg_out_deg(N, sigmas[i], boost[j], delta, q)
            var_grid = hogn_var_out_deg(N, sigmas[i], boost[j], delta, q)
            avg_arr.append(np.sum(avg_grid)/N)
            var_arr.append(np.sqrt(np.sum(var_grid))/N)
        ax[i].plot(boost, avg_arr, color='red')
        ax2 = ax[i].twinx()
        ax2.plot(boost, np.sqrt(var_arr), color='blue')
        ax2.set_ylabel(r'$\sqrt{\mathrm{Var}(N_{ij})}$', color='blue')
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax[i].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    ax[0].set_title(r'$\sigma=1$')
    ax[0].set_xlabel(r'$\rho$')
    ax[0].set_ylabel(r'$\langle N_{ij} \rangle/N$', color='red')
    ax[1].set_title(r'$\sigma=\sqrt{N}/2$')
    ax[1].set_xlabel(r'$\rho$')
    ax[1].set_ylabel(r'$\langle N_{ij} \rangle/N$', color='red')
    ax[2].set_title(r'$\sigma=\sqrt{2N}$')
    ax[2].set_ylabel(r'$\langle N_{ij} \rangle/N$', color='red')
    ax[2].set_xlabel(r'$\rho$')
    plt.tight_layout()
    plt.show()


# def fig_4(N):
#
#     """
#     Entropy of a homogeneous connectivity matrix
#     """
#
#     M = int(round(np.sqrt(N)))
#     mat = np.zeros((100,100))
#     delta = 1
#
#     #get a vector of distances (which are the same for every neuron)
#     xv, yv = np.meshgrid(np.arange(M),np.arange(M))
#     X, Y = xv.ravel(), yv.ravel()
#     d_vec = np.array([dist((0,0),(X[i],Y[i]),M,delta) for i in range(N)])[1:]
#     sigmas = np.linspace(1, 100, 100)
#
#     for i, sigma in enumerate(sigmas):
#         rho_max = sigma*np.sqrt(2*np.pi)*np.exp(delta**2/(2*sigma**2))
#         rhos = np.linspace(0, rho_max*0.5, 100)
#         for j, rho in enumerate(rhos):
#             p_ij_vec, q_vec = kern(d_vec, sigma, rho)
#             H_ij_vec = entropy(p_ij_vec, p_ij_vec, q_vec)
#             H_total = 0.5*np.sum(H_ij_vec)
#             mat[i,j] = H_total
#
#     fig, ax = plt.subplots(1,2, figsize=(7,3))
#     colors = cm.coolwarm(np.linspace(0,1,100))
#     for i in range(mat.shape[0]):
#         ax[1].plot(mat[i,:], color=colors[i])
#     ax[1].set_xticks([0,100])
#     ax[1].set_xticklabels(['0', r'$\rho_{max}$'])
#     ax[1].set_ylabel(r'$H_{\sigma}(\rho) \;\;\;[bits]$')
#     norm = mpl.colors.Normalize(vmin=sigmas.min(), vmax=sigmas.max())
#     map = mpl.cm.ScalarMappable(norm=norm, cmap='coolwarm')
#     plt.colorbar(map, ax=ax[1], label=r'$\mathbf{\sigma}$', fraction=0.046, pad=0.04)
#
#     p_ij = np.linspace(0,1)
#     p_ji = np.linspace(0,1)
#     q = (1-p_ij)**2
#     z = p_ij+p_ji+q
#     ax[0].plot(p_ij, entropy(p_ij/z,p_ji/z,q/z), color='blue')
#     ax[0].set_xlabel(r'$p_{ij}$')
#     ax[0].set_ylabel(r'$H_{3}(p_{ij}) \;\;\; [bits]$')
#     plt.tight_layout()
#     plt.show()
