import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import cm
from hebb.util import *
from hebb.models import *
from matplotlib.ticker import FormatStrFormatter

def fig_1():


    """
    Generate a homogeneous gaussian network and plot the graph in spring format
    """

    N = 100
    M = int(round(np.sqrt(N)))
    #Define parameter maps
    q = 0.1; sigma = 5*np.ones((M,M))
    net = GaussianNetwork(N, sigma, q)
    custom_lines = [Line2D([0],[0],color='salmon', lw=4),Line2D([0],[0],color='cornflowerblue', lw=4)]

    fig = plt.figure(figsize=(8,3.5))
    gs = fig.add_gridspec(2,4, wspace=0.75, hspace=0.75)
    ax0 = fig.add_subplot(gs[:, :2])
    ax1 = fig.add_subplot(gs[0, 2:3])
    ax2 = fig.add_subplot(gs[0, 3:4])
    ax3 = fig.add_subplot(gs[1, 2:3])
    ax4 = fig.add_subplot(gs[1, 3:4])

    # add_spring_graph(ax0, net, alpha=0.03)
    add_ego_graph(ax0, net)
    ax0.legend(custom_lines, ['In', 'Out'], loc='upper right')

    """
    Plot theoretical mean out degree as a function of sigma parameter with fixed sparsity
    """

    sigmas = np.linspace(np.sqrt(N)/16, np.sqrt(N)/2, 100)
    avg_arr, var_arr = hogn_out_deg_fixq(N, sigmas, 0.2)
    ax1.plot(sigmas, avg_arr, color='red')
    ax1.set_xlabel(r'$\sigma$')
    ax1.set_xticks([sigmas.min(), sigmas.max()])
    ax1.set_xticklabels([r'$\sqrt{N}/16$',r'$\sqrt{N}/2$'])
    ax1.set_ylabel(r'$\langle N_{ij} \rangle/N$')
    ax1.set_title(r'$q=0.2$')


    """
    Generate a map of mean out degree over the 2D parameter space
    """

    qs = np.linspace(0,1,100)
    avg_arr, var_arr = hogn_out_deg_full(N, sigmas, qs)
    ax2.imshow(avg_arr, cmap='coolwarm', origin='lower')
    ax2.set_xlabel(r'$\sigma$')
    ax2.set_xticks([0,100])
    ax2.set_xticklabels([r'$\sqrt{N}/16$',r'$\sqrt{N}/2$'])
    ax2.set_yticks([0,100])
    ax2.set_yticklabels([0,1])
    ax2.set_ylabel(r'$q$')
    ax3.set_title(r'$\sigma=\sqrt{N}/8$')
    ax4.set_title(r'$\sigma=\sqrt{N}/2$')
    colormap = cm.get_cmap('coolwarm')
    norm = mpl.colors.Normalize(vmin=0, vmax=avg_arr.max())
    map = mpl.cm.ScalarMappable(norm=norm, cmap=colormap)
    plt.colorbar(map, ax=ax2, fraction=0.046, pad=0.04, label=r'$\langle N_{ij} \rangle/N$')


    """
    Plot theoretical mean out degree as a function of sparsity parameter with fixed sigma
    """

    w = 20
    qs = np.linspace(0.1,1,w)
    ax = [ax3,ax4]
    sigmas = np.array([np.sqrt(N)/8,np.sqrt(N)/2])
    for i, sigma in enumerate(sigmas):
        avg_arr, var_arr = hogn_out_deg_fixsig(N, sigma, qs)
        ax[i].plot(qs, avg_arr/N, color='blue')
        ax[i].set_xlabel(r'$q$')
        ax[i].set_ylabel(r'$\langle N_{ij} \rangle/N$')

    """
    Plot observed mean out degree as a function of sparsity parameter with fixed sigma
    """

    w = 20
    qs = np.linspace(0.1,1,w)
    ax = [ax3,ax4]
    sig1 = np.ones((M,M))*np.sqrt(N)/8
    sig2 = np.ones((M,M))*np.sqrt(N)/2
    n_ij_sig1 = np.zeros_like(qs)
    n_ij_sig2 = np.zeros_like(qs)
    for i,q in enumerate(qs):
        n_ij_sig1[i] = np.mean(np.sum(GaussianNetwork(N, sig1, q).C,axis=0))
        n_ij_sig2[i] = np.mean(np.sum(GaussianNetwork(N, sig2, q).C,axis=0))
    ax3.plot(qs, n_ij_sig1/N, color='cyan', linestyle='--')
    ax4.plot(qs, n_ij_sig2/N, color='cyan', linestyle='--')
    plt.tight_layout()

def fig_2():


    """
    Generate a heterogeneous gaussian network and plot the graph in spring format
    """

    N = 100
    M = int(round(np.sqrt(N)))
    q = 0.1
    sigma_e = 3; sigma_i = 5
    bias_e = 0.5; bias_i = 1
    net = ExInGaussianNetwork(N, sigma_e, sigma_i, bias_e, bias_i, q, p_e=0.8)
    custom_lines = [Line2D([0],[0],color='salmon', lw=4),Line2D([0],[0],color='cornflowerblue', lw=4)]

    fig = plt.figure(figsize=(8,3.5))
    gs = fig.add_gridspec(2,4, wspace=0.75, hspace=0.75)
    ax0 = fig.add_subplot(gs[:, :2])
    ax1 = fig.add_subplot(gs[0, 2:3])
    ax2 = fig.add_subplot(gs[0, 3:4])
    ax3 = fig.add_subplot(gs[1, 2:4])

    add_spring_graph(ax0, net)


    """
    Fix q, p_e, bias_e, and bias_i and generate <N_ij^E> and <N_ij^I>
    maps of sigma_e vs sigma_i
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
