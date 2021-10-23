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
    Generate a excitatory-inhibitory gaussian network with excitatory
    and inhibitory bias parameters equal to one.

    This figure is for validating the numerical solution with the experiemental
    """

    N = 100
    M = int(round(np.sqrt(N)))
    q = 0.9
    p_e = 0.5
    sigma_e = 5; sigma_i = 5
    bias_e = 1; bias_i = 1
    net = ExInGaussianNetwork(N, sigma_e, sigma_i, bias_e, bias_i, q, p_e=p_e)
    custom_lines = [Line2D([0],[0],color='salmon', lw=4),Line2D([0],[0],color='cornflowerblue', lw=4)]

    fig = plt.figure(figsize=(12,4))
    gs = fig.add_gridspec(2,6, wspace=1, hspace=1)
    ax0 = fig.add_subplot(gs[:, :2])
    ax1 = fig.add_subplot(gs[0, 2:3])
    ax2 = fig.add_subplot(gs[0, 3:4])
    ax3 = fig.add_subplot(gs[1, 2:3])
    ax4 = fig.add_subplot(gs[1, 3:4])
    ax5 = fig.add_subplot(gs[0, 4:5])
    #ax6 = fig.add_subplot(gs[0, 5:6])
    ax7 = fig.add_subplot(gs[1, 4:5])
    #ax8 = fig.add_subplot(gs[1, 5:6])

    add_spring_graph(ax0, net)


    """
    Fix q, p_e, bias_e, and bias_i and generate maps of sigma_e vs sigma_i
    """

    nsigma = 10
    sigmas = np.linspace(np.sqrt(N)/16,np.sqrt(N)/2,nsigma)

    n_ee_out, n_ee_in, n_ei_out, n_ei_in =\
    exin_e_deg_fixqbias(N, sigmas, bias_e, bias_i, q, p_e)
    n_ee_out /= N*p_e
    n_ee_in /= N*p_e
    n_ei_out /= N*p_e
    n_ei_in /= N*(1-p_e) #I -> E

    n_ii_out, n_ii_in, n_ie_out, n_ie_in =\
    exin_i_deg_fixqbias(N, sigmas, bias_e, bias_i, q, p_e)
    n_ii_out /= N*(1-p_e)
    n_ii_in /= N*(1-p_e)
    n_ie_out /= N*(1-p_e)
    n_ie_in /= N*p_e

    ax1.plot(np.mean(n_ii_in,axis=0), color='red')
    ax1.set_xlabel(r'$\sigma_{I}$')
    ax1.set_ylabel(r'$\langle I_{out}^{I}\rangle, \langle I_{in}^{I}\rangle$')

    ax2.imshow(n_ei_in, origin='lower', cmap='coolwarm')
    ax2.set_xlabel(r'$\sigma_{I}$')
    ax2.set_ylabel(r'$\sigma_{E}$')
    ax2.set_xticks([0,nsigma])
    ax2.set_xticklabels([r'$\sqrt{N}/16$', r'$\sqrt{N}/2$'])
    ax2.set_yticks([0,nsigma])
    ax2.set_yticklabels([r'$\sqrt{N}/16$', r'$\sqrt{N}/2$'])
    ax2.xaxis.tick_top()
    ax2.xaxis.set_label_position('top')
    colormap = cm.get_cmap('coolwarm')
    norm = mpl.colors.Normalize(vmin=0, vmax=n_ei_in.max())
    map = mpl.cm.ScalarMappable(norm=norm, cmap=colormap)
    plt.colorbar(map, ax=ax2, fraction=0.046, pad=0.04, orientation='horizontal', label=r'$\langle E_{in}^{I}\rangle,\langle I_{out}^{E}\rangle$')

    ax3.plot(np.mean(n_ee_in,axis=1), color='red')
    ax3.set_xlabel(r'$\sigma_{E}$')
    ax3.set_ylabel(r'$\sigma_{E}$')
    ax3.set_ylabel(r'$\langle E_{out}^{E}\rangle$, $\langle E_{in}^{E}\rangle$')


    ax4.imshow(n_ei_out, origin='lower', cmap='coolwarm')
    ax4.set_xlabel(r'$\sigma_{I}$')
    ax4.set_ylabel(r'$\sigma_{E}$')
    ax4.set_xticks([0,nsigma])
    ax4.set_xticklabels([r'$\sqrt{N}/16$', r'$\sqrt{N}/2$'])
    ax4.set_yticks([0,nsigma])
    ax4.set_yticklabels([r'$\sqrt{N}/16$', r'$\sqrt{N}/2$'])
    ax4.xaxis.tick_top()
    ax4.xaxis.set_label_position('top')
    colormap = cm.get_cmap('coolwarm')
    norm = mpl.colors.Normalize(vmin=0, vmax=n_ei_out.max())
    map = mpl.cm.ScalarMappable(norm=norm, cmap=colormap)
    plt.colorbar(map, ax=ax4, fraction=0.046, pad=0.04, orientation='horizontal', label=r'$\langle E_{out}^{I}\rangle,\langle I_{in}^{E}\rangle$')

    """
    Generate several networks to test numerical predictions
    """

    arr = []
    xv, yv = np.meshgrid(np.arange(nsigma),np.arange(nsigma))
    X, Y = xv.ravel(), yv.ravel()
    sig_e_v, sig_i_v = np.meshgrid(sigmas,sigmas)
    sigma_e, sigma_i = sig_e_v.ravel(), sig_i_v.ravel()
    ee_mat = np.zeros((nsigma,nsigma))
    ii_mat = np.zeros((nsigma,nsigma))
    ei_mat = np.zeros((nsigma,nsigma))
    ie_mat = np.zeros((nsigma,nsigma))
    for i in range(sigma_e.shape[0]):
        net = ExInGaussianNetwork(N, sigma_e[i], sigma_i[i], bias_e, bias_i, q, p_e=p_e, delta=1)
        ei_avg, ie_avg, ee_avg, ii_avg = net.get_deg_avg()
        ei_mat[X[i],Y[i]] = ei_avg/(N*(1-p_e))
        ie_mat[X[i],Y[i]] = ie_avg/(N*p_e)
        ee_mat[X[i],Y[i]] = ee_avg/(N*p_e)
        ii_mat[X[i],Y[i]] = ii_avg/(N*(1-p_e))

    ax1.plot(np.mean(ii_mat,axis=0), color='blue')
    ax3.plot(np.mean(ee_mat,axis=1), color='blue')

    ax5.imshow(ie_mat, origin='lower', cmap='coolwarm')
    ax5.set_xlabel(r'$\sigma_{I}$')
    ax5.set_ylabel(r'$\sigma_{E}$')
    ax5.set_xticks([0,nsigma])
    ax5.set_xticklabels([r'$\sqrt{N}/16$', r'$\sqrt{N}/2$'])
    ax5.set_yticks([0,nsigma])
    ax5.set_yticklabels([r'$\sqrt{N}/16$', r'$\sqrt{N}/2$'])
    ax5.xaxis.tick_top()
    ax5.xaxis.set_label_position('top')
    colormap = cm.get_cmap('coolwarm')
    norm = mpl.colors.Normalize(vmin=0, vmax=ie_mat.max())
    map = mpl.cm.ScalarMappable(norm=norm, cmap=colormap)
    plt.colorbar(map, ax=ax5, fraction=0.046, pad=0.04, orientation='horizontal', label=r'$\langle E_{in}^{I}\rangle,\langle I_{out}^{E}\rangle$')

    ax7.imshow(ei_mat, origin='lower', cmap='coolwarm')
    ax7.set_xlabel(r'$\sigma_{I}$')
    ax7.set_ylabel(r'$\sigma_{E}$')
    ax7.set_xticks([0,nsigma])
    ax7.set_xticklabels([r'$\sqrt{N}/16$', r'$\sqrt{N}/2$'])
    ax7.set_yticks([0,nsigma])
    ax7.set_yticklabels([r'$\sqrt{N}/16$', r'$\sqrt{N}/2$'])
    ax7.xaxis.tick_top()
    ax7.xaxis.set_label_position('top')
    colormap = cm.get_cmap('coolwarm')
    norm = mpl.colors.Normalize(vmin=0, vmax=ei_mat.max())
    map = mpl.cm.ScalarMappable(norm=norm, cmap=colormap)
    plt.colorbar(map, ax=ax7, fraction=0.046, pad=0.04, orientation='horizontal', label=r'$\langle E_{out}^{I}\rangle,\langle I_{in}^{E}\rangle$')
    plt.tight_layout()

def fig_3():


    """
    Generate a excitatory-inhibitory gaussian network with excitatory
    and inhibitory bias parameters equal to one.

    This figure is for validating the numerical solution with the experiemental
    """

    fig = plt.figure(figsize=(7,8))
    gs = fig.add_gridspec(4,4, wspace=1, hspace=2)
    ax0 = fig.add_subplot(gs[:2, :2])
    ax1 = fig.add_subplot(gs[0, 2:3])
    ax2 = fig.add_subplot(gs[0, 3:4])
    ax3 = fig.add_subplot(gs[1, 2:3])
    ax4 = fig.add_subplot(gs[1, 3:4])

    ax5 = fig.add_subplot(gs[2:, :2])
    ax6 = fig.add_subplot(gs[2, 2:3])
    ax7 = fig.add_subplot(gs[2, 3:4])
    ax8 = fig.add_subplot(gs[3, 2:3])
    ax9 = fig.add_subplot(gs[3, 3:4])

    N = 100
    M = int(round(np.sqrt(N)))
    q = 0.1
    p_e = 0.5
    sigma_e = 5; sigma_i = 5
    bias_e = 1; bias_i = 1
    net = ExInGaussianNetwork(N, sigma_e, sigma_i, bias_e, bias_i, q, p_e=p_e)
    custom_lines = [Line2D([0],[0],color='salmon', lw=4),Line2D([0],[0],color='cornflowerblue', lw=4)]
    add_spring_graph(ax0, net)


    """
    Fix q, p_e, bias_e, and bias_i and generate maps of sigma_e vs sigma_i
    """

    nsigma = 20
    sigmas = np.linspace(np.sqrt(N)/16,np.sqrt(N)/2,nsigma)

    n_ee_out, n_ee_in, n_ei_out, n_ei_in =\
    exin_e_deg_fixqbias(N, sigmas, bias_e, bias_i, q, p_e)
    n_ee_out /= N*p_e
    n_ee_in /= N*p_e
    n_ei_out /= N*p_e
    n_ei_in /= N*(1-p_e) #I -> E

    n_ii_out, n_ii_in, n_ie_out, n_ie_in =\
    exin_i_deg_fixqbias(N, sigmas, bias_e, bias_i, q, p_e)
    n_ii_out /= N*(1-p_e)
    n_ii_in /= N*(1-p_e)
    n_ie_out /= N*(1-p_e)
    n_ie_in /= N*p_e

    ax1.plot(np.mean(n_ii_in,axis=0), color='red')
    ax1.set_xlabel(r'$\sigma_{I}$')
    ax1.set_ylabel(r'$\langle I_{out}^{I}\rangle, \langle I_{in}^{I}\rangle$')

    ax2.imshow(n_ei_in, origin='lower', cmap='coolwarm')
    ax2.set_xlabel(r'$\sigma_{I}$')
    ax2.set_ylabel(r'$\sigma_{E}$')
    ax2.set_xticks([0,nsigma])
    ax2.set_xticklabels([r'$\sqrt{N}/16$', r'$\sqrt{N}/2$'])
    ax2.set_yticks([0,nsigma])
    ax2.set_yticklabels([r'$\sqrt{N}/16$', r'$\sqrt{N}/2$'])
    ax2.xaxis.tick_top()
    ax2.xaxis.set_label_position('top')
    colormap = cm.get_cmap('coolwarm')
    norm = mpl.colors.Normalize(vmin=0, vmax=n_ei_in.max())
    map = mpl.cm.ScalarMappable(norm=norm, cmap=colormap)
    plt.colorbar(map, ax=ax2, fraction=0.046, pad=0.04, orientation='horizontal', label=r'$\langle E_{in}^{I}\rangle,\langle I_{out}^{E}\rangle$')

    ax3.plot(np.mean(n_ee_in,axis=1), color='red')
    ax3.set_xlabel(r'$\sigma_{E}$')
    ax3.set_ylabel(r'$\sigma_{E}$')
    ax3.set_ylabel(r'$\langle E_{out}^{E}\rangle$, $\langle E_{in}^{E}\rangle$')


    ax4.imshow(n_ei_out, origin='lower', cmap='coolwarm')
    ax4.set_xlabel(r'$\sigma_{I}$')
    ax4.set_ylabel(r'$\sigma_{E}$')
    ax4.set_xticks([0,nsigma])
    ax4.set_xticklabels([r'$\sqrt{N}/16$', r'$\sqrt{N}/2$'])
    ax4.set_yticks([0,nsigma])
    ax4.set_yticklabels([r'$\sqrt{N}/16$', r'$\sqrt{N}/2$'])
    ax4.xaxis.tick_top()
    ax4.xaxis.set_label_position('top')
    colormap = cm.get_cmap('coolwarm')
    norm = mpl.colors.Normalize(vmin=0, vmax=n_ei_out.max())
    map = mpl.cm.ScalarMappable(norm=norm, cmap=colormap)
    plt.colorbar(map, ax=ax4, fraction=0.046, pad=0.04, orientation='horizontal', label=r'$\langle E_{out}^{I}\rangle,\langle I_{in}^{E}\rangle$')

    N = 100
    M = int(round(np.sqrt(N)))
    q = 0.9
    p_e = 0.5
    sigma_e = 5; sigma_i = 5
    bias_e = 1; bias_i = 1
    net = ExInGaussianNetwork(N, sigma_e, sigma_i, bias_e, bias_i, q, p_e=p_e)
    custom_lines = [Line2D([0],[0],color='salmon', lw=4),Line2D([0],[0],color='cornflowerblue', lw=4)]
    add_spring_graph(ax5, net)


    """
    Fix q, p_e, bias_e, and bias_i and generate maps of sigma_e vs sigma_i
    """

    nsigma = 20
    sigmas = np.linspace(np.sqrt(N)/16,np.sqrt(N)/2,nsigma)

    n_ee_out, n_ee_in, n_ei_out, n_ei_in =\
    exin_e_deg_fixqbias(N, sigmas, bias_e, bias_i, q, p_e)
    n_ee_out /= N*p_e
    n_ee_in /= N*p_e
    n_ei_out /= N*p_e
    n_ei_in /= N*(1-p_e) #I -> E

    n_ii_out, n_ii_in, n_ie_out, n_ie_in =\
    exin_i_deg_fixqbias(N, sigmas, bias_e, bias_i, q, p_e)
    n_ii_out /= N*(1-p_e)
    n_ii_in /= N*(1-p_e)
    n_ie_out /= N*(1-p_e)
    n_ie_in /= N*p_e

    ax6.plot(np.mean(n_ii_in,axis=0), color='red')
    ax6.set_xlabel(r'$\sigma_{I}$')
    ax6.set_ylabel(r'$\langle I_{out}^{I}\rangle, \langle I_{in}^{I}\rangle$')

    ax7.imshow(n_ei_in, origin='lower', cmap='coolwarm')
    ax7.set_xlabel(r'$\sigma_{I}$')
    ax7.set_ylabel(r'$\sigma_{E}$')
    ax7.set_xticks([0,nsigma])
    ax7.set_xticklabels([r'$\sqrt{N}/16$', r'$\sqrt{N}/2$'])
    ax7.set_yticks([0,nsigma])
    ax7.set_yticklabels([r'$\sqrt{N}/16$', r'$\sqrt{N}/2$'])
    ax7.xaxis.tick_top()
    ax7.xaxis.set_label_position('top')
    colormap = cm.get_cmap('coolwarm')
    norm = mpl.colors.Normalize(vmin=0, vmax=n_ei_in.max())
    map = mpl.cm.ScalarMappable(norm=norm, cmap=colormap)
    plt.colorbar(map, ax=ax7, fraction=0.046, pad=0.04, orientation='horizontal', label=r'$\langle E_{in}^{I}\rangle,\langle I_{out}^{E}\rangle$')

    ax8.plot(np.mean(n_ee_in,axis=1), color='red')
    ax8.set_xlabel(r'$\sigma_{E}$')
    ax8.set_ylabel(r'$\sigma_{E}$')
    ax8.set_ylabel(r'$\langle E_{out}^{E}\rangle$, $\langle E_{in}^{E}\rangle$')


    ax9.imshow(n_ei_out, origin='lower', cmap='coolwarm')
    ax9.set_xlabel(r'$\sigma_{I}$')
    ax9.set_ylabel(r'$\sigma_{E}$')
    ax9.set_xticks([0,nsigma])
    ax9.set_xticklabels([r'$\sqrt{N}/16$', r'$\sqrt{N}/2$'])
    ax9.set_yticks([0,nsigma])
    ax9.set_yticklabels([r'$\sqrt{N}/16$', r'$\sqrt{N}/2$'])
    ax9.xaxis.tick_top()
    ax9.xaxis.set_label_position('top')
    colormap = cm.get_cmap('coolwarm')
    norm = mpl.colors.Normalize(vmin=0, vmax=n_ei_out.max())
    map = mpl.cm.ScalarMappable(norm=norm, cmap=colormap)
    plt.colorbar(map, ax=ax9, fraction=0.046, pad=0.04, orientation='horizontal', label=r'$\langle E_{out}^{I}\rangle,\langle I_{in}^{E}\rangle$')


    # """
    # Generate several networks to test numerical predictions
    # """
    #
    # arr = []
    # xv, yv = np.meshgrid(np.arange(nsigma),np.arange(nsigma))
    # X, Y = xv.ravel(), yv.ravel()
    # sig_e_v, sig_i_v = np.meshgrid(sigmas,sigmas)
    # sigma_e, sigma_i = sig_e_v.ravel(), sig_i_v.ravel()
    # ee_mat = np.zeros((nsigma,nsigma))
    # ii_mat = np.zeros((nsigma,nsigma))
    # ei_mat = np.zeros((nsigma,nsigma))
    # ie_mat = np.zeros((nsigma,nsigma))
    # for i in range(sigma_e.shape[0]):
    #     net = ExInGaussianNetwork(N, sigma_e[i], sigma_i[i], bias_e, bias_i, q, p_e=p_e, delta=1)
    #     ei_avg, ie_avg, ee_avg, ii_avg = net.get_deg_avg()
    #     ei_mat[X[i],Y[i]] = ei_avg/(N*(1-p_e))
    #     ie_mat[X[i],Y[i]] = ie_avg/(N*p_e)
    #     ee_mat[X[i],Y[i]] = ee_avg/(N*p_e)
    #     ii_mat[X[i],Y[i]] = ii_avg/(N*(1-p_e))
    #
    # ax1.plot(np.mean(ii_mat,axis=0), color='blue')
    # ax3.plot(np.mean(ee_mat,axis=1), color='blue')
    #
    # ax5.imshow(ie_mat, origin='lower', cmap='coolwarm')
    # ax5.set_xlabel(r'$\sigma_{I}$')
    # ax5.set_ylabel(r'$\sigma_{E}$')
    # ax5.set_xticks([0,nsigma])
    # ax5.set_xticklabels([r'$\sqrt{N}/16$', r'$\sqrt{N}/2$'])
    # ax5.set_yticks([0,nsigma])
    # ax5.set_yticklabels([r'$\sqrt{N}/16$', r'$\sqrt{N}/2$'])
    # ax5.xaxis.tick_top()
    # ax5.xaxis.set_label_position('top')
    # colormap = cm.get_cmap('coolwarm')
    # norm = mpl.colors.Normalize(vmin=0, vmax=ie_mat.max())
    # map = mpl.cm.ScalarMappable(norm=norm, cmap=colormap)
    # plt.colorbar(map, ax=ax5, fraction=0.046, pad=0.04, orientation='horizontal', label=r'$\langle E_{in}^{I}\rangle,\langle I_{out}^{E}\rangle$')
    #
    # ax7.imshow(ei_mat, origin='lower', cmap='coolwarm')
    # ax7.set_xlabel(r'$\sigma_{I}$')
    # ax7.set_ylabel(r'$\sigma_{E}$')
    # ax7.set_xticks([0,nsigma])
    # ax7.set_xticklabels([r'$\sqrt{N}/16$', r'$\sqrt{N}/2$'])
    # ax7.set_yticks([0,nsigma])
    # ax7.set_yticklabels([r'$\sqrt{N}/16$', r'$\sqrt{N}/2$'])
    # ax7.xaxis.tick_top()
    # ax7.xaxis.set_label_position('top')
    # colormap = cm.get_cmap('coolwarm')
    # norm = mpl.colors.Normalize(vmin=0, vmax=ei_mat.max())
    # map = mpl.cm.ScalarMappable(norm=norm, cmap=colormap)
    # plt.colorbar(map, ax=ax7, fraction=0.046, pad=0.04, orientation='horizontal', label=r'$\langle E_{out}^{I}\rangle,\langle I_{in}^{E}\rangle$')
    # plt.tight_layout()

def fig_4(lif, net, spikes, focal=0):

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
