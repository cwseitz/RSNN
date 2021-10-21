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

    N = 100
    M = int(round(np.sqrt(N)))
    #Define parameter maps
    q = 0.1; sigma = 5*np.ones((M,M))
    bias_ij = bias_ji = np.ones((M,M))
    net = GaussianNetwork(N, sigma, bias_ij, bias_ji, q)
    custom_lines = [Line2D([0],[0],color='salmon', lw=4),Line2D([0],[0],color='cornflowerblue', lw=4)]

    fig, ax_ = plt.subplots(1, 1, figsize=(7,4))
    ax_.set_xticks([]); ax_.set_yticks([])
    ax_.spines['right'].set_visible(False)
    ax_.spines['top'].set_visible(False)
    ax_.spines['left'].set_visible(False)
    ax_.spines['bottom'].set_visible(False)
    ax0 = ax_.inset_axes([0, 0, 0.45, 0.85])
    ax1 = ax_.inset_axes([0.6, 0.55, 0.2, 0.3])
    ax2 = ax_.inset_axes([0.9, 0.55, 0.2, 0.3])
    ax3 = ax_.inset_axes([0.6, 0, 0.2, 0.3])
    ax4 = ax_.inset_axes([0.9, 0, 0.2, 0.3])

    add_spring_graph(ax0, net, alpha=0.03)
    add_ego_graph(ax1, net)

    """
    Generate a map of mean out degree over the 2D parameter space
    """

    sigmas = np.linspace(np.sqrt(N)/16, np.sqrt(N)/2, 100)
    qs = np.linspace(0,1,100)
    avg_arr, var_arr = hom_out_deg_full(N, sigmas, qs)
    ax2.imshow(avg_arr, cmap='coolwarm', origin='lower')
    ax2.set_xlabel(r'$\sigma$')
    ax2.set_xticks([0,100])
    ax2.set_xticklabels([r'$\sqrt{N}/16$',r'$\sqrt{N}/2$'])
    ax2.set_yticks([0,100])
    ax2.set_yticklabels([0,1])
    ax2.set_ylabel(r'$q$')

    ax3.set_title(r'$\sigma=\sqrt{N}/8$')
    ax4.set_title(r'$\sigma=\sqrt{N}/2$')

    """
    Plot mean out degree as a function of sparsity parameter with fixed sigma
    """

    w = 20
    qs = np.linspace(0.1,1,w)
    ax = [ax3,ax4]
    sigmas = np.array([np.sqrt(N)/8,np.sqrt(N)/2])
    for i, sigma in enumerate(sigmas):
        avg_arr, var_arr = hom_out_deg_fixsig(N, sigma, qs)
        ax[i].plot(qs, avg_arr/N, color='blue')
        ax[i].set_xlabel(r'$q$')
        ax[i].set_ylabel(r'$\langle N_{ij} \rangle/N$')

    plt.tight_layout()

def fig_2():


    """
    Generate a heterogeneous gaussian network and plot the graph in spectral format
    """

    N = 100
    M = int(round(np.sqrt(N)))
    #Define parameter maps
    q = 0.1; sigma = 5*np.ones((M,M))
    bias_ij = bias_ji = np.ones((M,M))
    net = GaussianNetwork(N, sigma, bias_ij, bias_ji, q)
    custom_lines = [Line2D([0],[0],color='salmon', lw=4),Line2D([0],[0],color='cornflowerblue', lw=4)]

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax0 = ax.inset_axes([0, 0, 0.5, 1.0])
    ax1 = ax.inset_axes([0.65, 0.55, 0.2, 0.4])
    ax2 = ax.inset_axes([1.05, 0.55, 0.2, 0.4])
    ax3 = ax.inset_axes([1.45, 0.55, 0.2, 0.4])
    ax4 = ax.inset_axes([0.65, 0, 0.2, 0.4])
    ax5 = ax.inset_axes([1.05, 0, 0.2, 0.4])
    ax6 = ax.inset_axes([1.45, 0, 0.2, 0.4])

    add_spring_graph(ax0, net)

    """
    Fix the sparsity parameter and get mean out_deg as a function of bias for a few sigma
    """

    sigmas = np.array([np.sqrt(N)/4,np.sqrt(N)/1, np.sqrt(N)])
    avg_arr, var_arr = out_deg_iters_fixq(N, sigmas, q)

    ax1.plot(avg_arr[0,:]/N, color='red')
    ax1.set_xlabel(r'$\gamma$')
    ax1.set_xticks([0,avg_arr.shape[1]])
    ax1.set_xticklabels([0,r'$\gamma_{max}$'])
    ax1.set_ylabel(r'$\langle N_{ij} \rangle/N$')
    ax1.set_title(r'$\sigma=\sqrt{N}/4$')
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    ax2.plot(avg_arr[1,:]/N, color='red')
    ax2.set_xlabel(r'$\gamma$')
    ax2.set_xticks([0,avg_arr.shape[1]])
    ax2.set_xticklabels([0,r'$\gamma_{max}$'])
    ax2.set_ylabel(r'$\langle N_{ij} \rangle/N$')
    ax2.set_title(r'$\sigma=\sqrt{N}/2$')
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    ax3.plot(avg_arr[2,:]/N, color='red')
    ax3.set_ylabel(r'$\langle N_{ij} \rangle/N$')
    ax3.set_xlabel(r'$\gamma$')
    ax3.set_xticks([0,avg_arr.shape[1]])
    ax3.set_xticklabels([0,r'$\gamma_{max}$'])
    ax3.set_title(r'$\sigma=\sqrt{N}/1$')
    ax3.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    """
    Generate parameter maps of (q, bias) with fixed sigma
    """

    w = 20
    qs = np.linspace(0.1,1,w)
    ax = [ax4,ax5,ax6]
    for i, sigma in enumerate(sigmas):
        avg_arr, var_arr = het_out_deg_fixsig(N, sigma, qs, n_bias=w)
        ax[i].imshow(avg_arr, origin='lower', cmap='coolwarm')
        ax[i].set_xticks([0,w])
        ax[i].set_ylabel(r'$q$')
        ax[i].set_yticklabels([0,1])
        ax[i].set_yticks([0,w])
        ax[i].set_xticklabels([1, r'$\gamma_{max}$'])
        ax[i].set_xlabel(r'$\gamma$')

    plt.tight_layout()

def fig_3(lif, net, spikes, focal=0):

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
