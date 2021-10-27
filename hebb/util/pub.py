import matplotlib as mpl
import matplotlib.pyplot as plt
import string
from matplotlib.lines import Line2D
from matplotlib import cm
from skimage.io import imread
from hebb.util import *
from hebb.models import *
from matplotlib.ticker import FormatStrFormatter
from .format_ax import *
mpl.rcParams['text.usetex'] = True

def fig_0():

    x = np.linspace(-5, 5, 100)
    x1, x2 = 0, 0
    sigma1, sigma2 = 1, 1
    y1 = np.exp(-(0.5*(x+x1)**2)/sigma1**2)
    y2 = np.exp(-(0.5*(x+x2)**2)/sigma2**2)
    y3 = 0.5*np.ones_like(x)

    fig, ax = plt.subplots(2,2, figsize=(6,5))
    ax[0,0].plot(x, y1, color='blue', label=r'$\Gamma_{ij}$')
    ax[0,0].plot(x, y2, color='red', linestyle='--', label=r'$\Gamma_{ji}$')
    #ax[0,0].plot(x, y3, color='cyan', label=r'$\Gamma_{0}$')
    ax[0,0].set_xlim([0,4])
    ax[0,0].set_xlabel(r'$|\Delta \mathbf{r}_{ij}|$', fontsize=12)
    ax[0,0].text(-0.1, 1.075, 'A', transform=ax[0,0].transAxes, size=16, weight='bold')
    format_ax(ax[0,0], ax_is_box=False)
    ax[0,0].set_xlabel(r'$|\Delta \mathbf{r}_{ij}|$', fontsize=12)
    ax[0,0].legend(loc='upper right')

    p12 = y1*(1-y2)*(1-y3)
    p21 = y2*(1-y1)*(1-y3)
    p0 = y3*(1-y1)*(1-y2)
    z = p12 + p21 + p0
    p12 /= z
    p21 /= z
    p0 /= z

    ax[0,1].plot(x, p12, color='blue', label=r'$p_{ij}$')
    ax[0,1].plot(x, p21, color='red', linestyle='--', label=r'$p_{ji}$')
    ax[0,1].plot(x, p0, color='cyan', label=r'$p_{0}$')
    ax[0,1].set_xlabel(r'$|\Delta \mathbf{r}_{ij}|$', fontsize=12)
    ax[0,1].set_xlim([0,4])
    format_ax(ax[0,1], ax_is_box=False)
    ax[0,1].set_xlabel(r'$|\Delta \mathbf{r}_{ij}|$', fontsize=12)
    ax[0,1].legend(loc='upper right')

    x = np.linspace(-5, 5, 100)
    x1, x2 = 0, 0
    sigma1, sigma2 = 2, 1
    y1 = np.exp(-(0.5*(x+x1)**2)/sigma1**2)
    y2 = np.exp(-(0.5*(x+x2)**2)/sigma2**2)
    y3 = 0.5*np.ones_like(x)


    ax[1,0].plot(x, y1, color='blue', label=r'$\Gamma_{ij}$')
    ax[1,0].plot(x, y2, color='red', linestyle='--', label=r'$\Gamma_{ji}$')
    #ax[1,0].plot(x, y3, color='cyan', label=r'$\Gamma_{0}$')
    ax[1,0].set_xlim([0,4])
    ax[1,0].set_xlabel(r'$|\Delta \mathbf{r}_{ij}|$', fontsize=12)
    ax[1,0].text(-0.1, 1.075, 'B', transform=ax[1,0].transAxes, size=16, weight='bold')
    format_ax(ax[1,0], ax_is_box=False)
    ax[1,0].set_xlabel(r'$|\Delta \mathbf{r}_{ij}|$', fontsize=12)
    ax[1,0].legend(loc='upper right')

    p12 = y1*(1-y2)*(1-y3)
    p21 = y2*(1-y1)*(1-y3)
    p0 = y3*(1-y1)*(1-y2)
    z = p12 + p21 + p0
    p12 /= z
    p21 /= z
    p0 /= z

    ax[1,1].plot(x, p12, color='blue', label=r'$p_{ij}$')
    ax[1,1].plot(x, p21, color='red', linestyle='--', label=r'$p_{ji}$')
    ax[1,1].plot(x, p0, color='cyan', label=r'$p_{0}$')
    ax[1,1].set_xlim([0,4])
    format_ax(ax[1,1], ax_is_box=False)
    ax[1,1].legend(loc='upper right')
    ax[1,1].set_xlabel(r'$|\Delta \mathbf{r}_{ij}|$', fontsize=12)
    plt.tight_layout()
    plt.show()

def fig_1(N,sigma,q):

    """
    Summarize the homogeneous gaussian network

    Parameters
    ----------
    N : int
        Number of neurons in the network. Must be a perfect square
    sigma : ndarray
        Standard deviation of the gaussian connectivity kernel
    q : ndarray
        Sparsity parameter

    """

    M = int(round(np.sqrt(N)))
    sigma = sigma*np.ones((M,M))
    net = GaussianNetwork(N, sigma, q)
    custom_lines = [Line2D([0],[0],color='red', lw=4),Line2D([0],[0],color='dodgerblue', lw=4)]

    fig = plt.figure(figsize=(9,6))
    gs = fig.add_gridspec(6,9, wspace=6, hspace=4)
    ax0 = fig.add_subplot(gs[:3, :3])
    ax1 = fig.add_subplot(gs[3:,:3])
    ax2 = fig.add_subplot(gs[:2, 4:6])
    ax3 = fig.add_subplot(gs[:2, 6:8])
    ax4 = fig.add_subplot(gs[2:4, 4:6])
    ax5 = fig.add_subplot(gs[2:4, 6:8])
    ax6 = fig.add_subplot(gs[4:6,4:6])
    ax7 = fig.add_subplot(gs[4:6,6:8])

    add_spring_graph(ax0, net)
    ax0.set_title(r'$N=100$')
    ax0.set_axis_off()
    ax0.text(-0.1, 0.9, 'A', transform=ax0.transAxes, size=10, weight='bold')

    add_ego_graph(ax1, net)
    ax1.set_axis_off()
    ax1.text(-0.1, 0.9, 'B', transform=ax1.transAxes, size=10, weight='bold')
    ax1.legend(custom_lines, ['In', 'Out'], loc='upper right')

    """
    Plot theoretical mean out degree as a function of sigma parameter with fixed sparsity
    """

    sigmas = np.linspace(np.sqrt(N)/16, np.sqrt(N)/2, 20)
    avg_arr, var_arr = gauss_net_deg_fixq(N, sigmas, q)
    ax2.plot(sigmas, avg_arr/N, color='red')

    format_ax(ax2,
              xlabel=r'$\sigma$',
              ylabel=r'$\langle N_{ij} \rangle/N$',
              xscale=[sigmas.min(), sigmas.max(), 1, None],
              ax_is_box=False)
    ax2.text(-0.2, 1.1, 'C', transform=ax2.transAxes, size=10, weight='bold')
    ax2.set_title(f'q={q}', fontsize=10)

    """
    Generate a map of mean out degree over the 2D parameter space
    """

    qs = np.linspace(0.1,0.9,20)
    avg_arr, var_arr = gauss_net_deg_full(N, sigmas, qs)
    ax3.imshow(avg_arr, cmap='coolwarm', origin='lower')
    ax3.text(-0.1, 1.1, 'D', transform=ax3.transAxes, size=10, weight='bold')
    format_ax(ax3,
              xlabel=r'$\sigma$',
              ylabel=r'$q$',
              ax_is_box=True)
    colormap = cm.get_cmap('coolwarm')
    norm = mpl.colors.Normalize(vmin=0, vmax=avg_arr.max())
    map = mpl.cm.ScalarMappable(norm=norm, cmap=colormap)
    plt.colorbar(map, ax=ax3, fraction=0.046, pad=0.04, label=r'$\langle N_{ij} \rangle/N$')


    """
    Plot theoretical mean out degree as a function of sparsity parameter with fixed sigma
    """

    w = 20
    qs = np.linspace(0.1,1,w)
    ax = [ax4,ax5]
    sigmas = np.array([np.sqrt(N)/8,np.sqrt(N)/2])
    for i, sigma in enumerate(sigmas):
        avg_arr, var_arr = gauss_net_deg_fixsig(N, sigma, qs)
        ax[i].plot(qs, avg_arr/N, color='blue')

    format_ax(ax4,
              xlabel=r'$q$',
              ylabel=r'$\langle N_{ij} \rangle/N$',
              xscale=[qs.min(), qs.max(), 0.5, None],
              ax_is_box=False)

    ax4.set_title(r'$\sigma=\sqrt{N}/8$', fontsize=10)
    ax4.text(-0.1, 1.1, 'E', transform=ax4.transAxes, size=10, weight='bold')

    """
    Plot observed mean out degree as a function of sparsity parameter with fixed sigma
    """

    w = 20
    qs = np.linspace(0.1,1,w)
    ax = [ax4,ax5]
    sig1 = np.ones((M,M))*np.sqrt(N)/8
    sig2 = np.ones((M,M))*np.sqrt(N)/2
    n_ij_sig1 = np.zeros_like(qs)
    n_ij_sig2 = np.zeros_like(qs)
    for i,q in enumerate(qs):
        n_ij_sig1[i] = np.mean(np.sum(GaussianNetwork(N, sig1, q).C,axis=0))
        n_ij_sig2[i] = np.mean(np.sum(GaussianNetwork(N, sig2, q).C,axis=0))
    ax4.plot(qs, n_ij_sig1/N, color='cyan', linestyle='--')
    ax5.plot(qs, n_ij_sig2/N, color='cyan', linestyle='--')

    format_ax(ax5,
              xlabel=r'$q$',
              ylabel=r'$\langle N_{ij} \rangle/N$',
              xscale=[qs.min(), qs.max(), 0.5, None],
              ax_is_box=False)

    ax5.set_title(r'$\sigma=\sqrt{N}/2$', fontsize=10)
    ax5.text(-0.1, 1.1, 'F', transform=ax5.transAxes, size=10, weight='bold')

    """
    Add single neuron kernels
    """

    add_kernel_pair(ax6, ax7, 1600, 20)
    format_ax(ax6)
    format_ax(ax7)
    ax6.set_xticks([]); ax6.set_yticks([])
    ax7.set_xticks([]); ax7.set_yticks([])
    ax6.text(-0.1, 1.1, 'G', transform=ax6.transAxes, size=10, weight='bold')
    ax7.text(-0.1, 1.1, 'H', transform=ax7.transAxes, size=10, weight='bold')
    colormap = cm.get_cmap('coolwarm')
    norm = mpl.colors.Normalize(vmin=0, vmax=2)
    map = mpl.cm.ScalarMappable(norm=norm, cmap=colormap)
    plt.colorbar(map, ax=ax6, orientation='horizontal', location='bottom', fraction=0.046, pad=0.04, label=r'$p_{ij}+p_{ji}$')

    colormap = cm.get_cmap('coolwarm')
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    map = mpl.cm.ScalarMappable(norm=norm, cmap=colormap)
    plt.colorbar(map, ax=ax7, orientation='horizontal', location='bottom', fraction=0.046, pad=0.04, label=r'$p_{ij}p_{ji}$')


    plt.tight_layout()

def fig_2():


    """
    Summarize shared connectivity in a homogeneous gaussian network

    Parameters
    ----------
    """

    fig, ax = plt.subplots(1,3, figsize=(10,3))

    N = 900; q = 0.8
    M = int(round(np.sqrt(N)))

    sigma = np.sqrt(N)/8
    net = GaussianNetwork(N, sigma*np.ones((M,M)), q)
    unique, avgs_arr = gauss_net_shared_exp(net) #experimental solution
    p_vec = gauss_net_shared(N, unique, sigma, q) #numerical solution
    ax[0].plot(unique, avgs_arr/N, color='red')
    ax[0].plot(unique, p_vec/N, color='blue')
    ax[0].text(-0.1, 1.075, 'A', transform=ax[0].transAxes, size=14, weight='bold')
    format_ax(ax[0],
              xlabel=r'$\Delta r_{ij}$',
              label_fontsize='large',
              ylabel=r'$\langle S_{ij}\rangle/N$',
              ax_is_box=False)
    ax[0].set_title(r'$\sigma=\sqrt{N}/8$', fontsize=14)

    sigma =  np.sqrt(N)/4
    net = GaussianNetwork(N, sigma*np.ones((M,M)), q)
    unique, avgs_arr = gauss_net_shared_exp(net) #experimental solution
    p_vec = gauss_net_shared(N, unique, sigma, q) #numerical solution
    ax[1].plot(unique, avgs_arr/N, color='red')
    ax[1].plot(unique, p_vec/N, color='blue')
    ax[1].text(-0.1, 1.075, 'B', transform=ax[1].transAxes, size=14, weight='bold')
    format_ax(ax[1],
              xlabel=r'$\Delta r_{ij}$',
              label_fontsize='large',
              ylabel=r'$\langle S_{ij}\rangle/N$',
              ax_is_box=False)
    ax[1].set_title(r'$\sigma=\sqrt{N}/4$', fontsize=14)

    sigma =  np.sqrt(N)/2
    net = GaussianNetwork(N, sigma*np.ones((M,M)), q)
    unique, avgs_arr = gauss_net_shared_exp(net) #experimental solution
    p_vec = gauss_net_shared(N, unique, sigma, q) #numerical solution
    ax[2].text(-0.1, 1.075, 'C', transform=ax[2].transAxes, size=14, weight='bold')
    ax[2].plot(unique, avgs_arr/N, color='red')
    ax[2].plot(unique, p_vec/N, color='blue')
    format_ax(ax[2],
              xlabel=r'$\Delta r_{ij}$',
              label_fontsize='large',
              ylabel=r'$\langle S_{ij}\rangle/N$',
              ax_is_box=False)
    ax[2].set_title(r'$\sigma=\sqrt{N}/2$', fontsize=14)
    plt.tight_layout()

def fig_3(N=100, q1=0.2, q2=0.8, sigma_e=5, sigma_i=5, p_e=0.8):


    """
    Summarize the excitatory-inhibitory gaussian network for two different
    values of the sparsity parameter q

    Parameters
    ----------
    N : int, optional
        Number of neurons in the network. Must be a perfect square
    q1 : ndarray, optional
        Sparsity parameter 1
    q2 : ndarray, optional
        Sparsity parameter 2
    sigma_e : float, optional
        Standard deviation of the excitatory kernel
    sigma_i : float, optional
        Standard deviation of the inhibitory kernel

    """

    fig = plt.figure(figsize=(7,10))
    gs = fig.add_gridspec(6,4, wspace=1, hspace=1)
    ax0 = fig.add_subplot(gs[:2, :])
    ax1 = fig.add_subplot(gs[2:4, :2])
    ax2 = fig.add_subplot(gs[2, 2:3])
    ax3 = fig.add_subplot(gs[2, 3:4])
    ax4 = fig.add_subplot(gs[3, 2:3])
    ax5 = fig.add_subplot(gs[3, 3:4])

    ax6 = fig.add_subplot(gs[4:, :2])
    ax7 = fig.add_subplot(gs[4, 2:3])
    ax8 = fig.add_subplot(gs[4, 3:4])
    ax9 = fig.add_subplot(gs[5, 2:3])
    ax10 = fig.add_subplot(gs[5, 3:4])

    ax0.set_axis_off()
    #ax0.text(0.1, 0.8, 'A', transform=ax0.transAxes, size=14, weight='bold')

    net = ExInGaussianNetwork(N, sigma_e, sigma_i, q1, p_e=p_e)
    custom_lines = [Line2D([0],[0],color='red', lw=4),Line2D([0],[0],color='dodgerblue', lw=4)]
    add_spring_graph(ax1, net)
    ax1.set_title(f'$q={q1}$')
    ax1.legend(custom_lines, ['E', 'I'], loc='upper left')
    ax1.set_axis_off()
    ax1.text(-0.1, 1.0, 'A', transform=ax1.transAxes, size=14, weight='bold')

    """
    Fix q, p_e, bias_e, and bias_i and generate maps of sigma_e vs sigma_i
    """

    nsigma = 20
    sigmas = np.linspace(np.sqrt(N)/16,np.sqrt(N)/2,nsigma)

    n_ee_out, n_ee_in, n_ei_out, n_ei_in =\
    exin_net_avg_edeg_fixq(N, sigmas, q1, p_e)
    n_ee_out /= N*p_e
    n_ee_in /= N*p_e
    n_ei_out /= N*p_e
    n_ei_in /= N*(1-p_e)

    n_ii_out, n_ii_in, n_ie_out, n_ie_in =\
    exin_net_avg_ideg_fixq(N, sigmas, q1, p_e)
    n_ii_out /= N*(1-p_e)
    n_ii_in /= N*(1-p_e)
    n_ie_out /= N*(1-p_e)
    n_ie_in /= N*p_e


    ax2.plot(np.mean(n_ii_in,axis=0), color='red')
    format_ax(ax2,
              xlabel=r'$\sigma_{I}$',
              ylabel=r'$\langle I_{I}\rangle/N_{I}$',
              ax_is_box=False)
    ax2.text(-0.4, 1.1, 'B', transform=ax2.transAxes, size=14, weight='bold')

    ax3.imshow(n_ei_in, origin='lower', cmap='coolwarm')
    format_ax(ax3,
              xlabel=r'$\sigma_{I}$',
              ylabel=r'$\sigma_{E}$')

    ax3.xaxis.tick_top()
    colormap = cm.get_cmap('coolwarm')
    norm = mpl.colors.Normalize(vmin=0, vmax=n_ei_in.max())
    map = mpl.cm.ScalarMappable(norm=norm, cmap=colormap)
    plt.colorbar(map, ax=ax3, fraction=0.046, pad=0.04, orientation='vertical', label=r'$\langle E_{I}\rangle$')

    ax4.plot(np.mean(n_ee_in,axis=1), color='red')
    format_ax(ax4,
              xlabel=r'$\sigma_{E}$',
              ylabel=r'$\langle E_{E}\rangle/N_{E}$',
              ax_is_box=False)


    ax5.imshow(n_ei_out, origin='lower', cmap='coolwarm')

    format_ax(ax5,
              xlabel=r'$\sigma_{I}$',
              ylabel=r'$\sigma_{E}$')

    ax5.xaxis.tick_top()

    colormap = cm.get_cmap('coolwarm')
    norm = mpl.colors.Normalize(vmin=0, vmax=n_ei_out.max())
    map = mpl.cm.ScalarMappable(norm=norm, cmap=colormap)
    plt.colorbar(map, ax=ax5, fraction=0.046, pad=0.04, orientation='vertical', label=r'$\langle I_{E}\rangle$')

    net = ExInGaussianNetwork(N, sigma_e, sigma_i, q2, p_e=p_e)
    custom_lines = [Line2D([0],[0],color='red', lw=4),Line2D([0],[0],color='dodgerblue', lw=4)]
    add_spring_graph(ax6, net)
    ax6.set_title(f'$q={q2}$')
    ax6.legend(custom_lines, ['E', 'I'], loc='upper left')
    ax6.set_axis_off()
    ax6.text(-0.1, 1.0, 'C', transform=ax6.transAxes, size=14, weight='bold')

    """
    Fix q, p_e, bias_e, and bias_i and generate maps of sigma_e vs sigma_i
    """

    nsigma = 20
    sigmas = np.linspace(np.sqrt(N)/16,np.sqrt(N)/2,nsigma)

    n_ee_out, n_ee_in, n_ei_out, n_ei_in =\
    exin_net_avg_edeg_fixq(N, sigmas, q2, p_e)
    n_ee_out /= N*p_e
    n_ee_in /= N*p_e
    n_ei_out /= N*p_e
    n_ei_in /= N*(1-p_e)

    n_ii_out, n_ii_in, n_ie_out, n_ie_in =\
    exin_net_avg_ideg_fixq(N, sigmas, q2, p_e)
    n_ii_out /= N*(1-p_e)
    n_ii_in /= N*(1-p_e)
    n_ie_out /= N*(1-p_e)
    n_ie_in /= N*p_e

    ax7.plot(np.mean(n_ii_in,axis=0), color='red')
    ax7.text(-0.3, 1.1, 'D', transform=ax7.transAxes, size=14, weight='bold')
    format_ax(ax7,
              xlabel=r'$\sigma_{I}$',
              ylabel=r'$\langle I_{I}\rangle/N_{I}$',
              ax_is_box=False)

    ax8.imshow(n_ei_in, origin='lower', cmap='coolwarm')
    format_ax(ax8,
              xlabel=r'$\sigma_{I}$',
              ylabel=r'$\sigma_{E}$')

    ax8.xaxis.tick_top()
    colormap = cm.get_cmap('coolwarm')
    norm = mpl.colors.Normalize(vmin=0, vmax=n_ei_in.max())
    map = mpl.cm.ScalarMappable(norm=norm, cmap=colormap)
    plt.colorbar(map, ax=ax8, fraction=0.046, pad=0.04, orientation='vertical', label=r'$\langle E_{I}\rangle$')

    ax9.plot(np.mean(n_ee_in,axis=1), color='red')

    format_ax(ax9,
              xlabel=r'$\sigma_{E}$',
              ylabel=r'$\langle E_{E}\rangle/N_{E}$',
              ax_is_box=False)


    ax10.imshow(n_ei_in, origin='lower', cmap='coolwarm')
    format_ax(ax10,
              xlabel=r'$\sigma_{I}$',
              ylabel=r'$\sigma_{E}$')
    ax10.xaxis.tick_top()
    colormap = cm.get_cmap('coolwarm')
    norm = mpl.colors.Normalize(vmin=0, vmax=n_ei_out.max())
    map = mpl.cm.ScalarMappable(norm=norm, cmap=colormap)
    plt.colorbar(map, ax=ax10, fraction=0.046, pad=0.04, orientation='vertical', label=r'$\langle I_{E}\rangle$')

def fig_4(N=400, sigma_e=5, sigma_i=5, q=0.8, p_e=0.8):

    """
    Shared connectivity in an excitatory-inhibitory gaussian network

    Parameters
    ----------
    N : int, optional
        Number of neurons in the network. Must be a perfect square
    sigma_e : float, optional
        Standard deviation of the excitatory kernel
    sigma_i : float, optional
        Standard deviation of the inhibitory kernel
    q : ndarray, optional
        Sparsity parameter

    """

    net = ExInGaussianNetwork(N, sigma_e, sigma_i, q, p_e=p_e)
    fig, ax = plt.subplots()

    ex_ex_dists, ex_ex_nshared = exin_net_shared_exp(net.C, net.ex_idx, net.ex_idx, net.M)
    ex_in_dists, ex_in_nshared = exin_net_shared_exp(net.C, net.in_idx, net.ex_idx, net.M)

    ax.plot(ex_ex_dists,ex_ex_nshared, color='red')
    ax.plot(ex_in_dists,ex_in_nshared, color='blue')

def fig_5(lif, net, spikes, focal=0):

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
