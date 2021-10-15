import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import cm

def fig_1(SN_2D):

    fig, ax = plt.subplots(1,2, figsize=(6,2.25))
    custom_lines = [Line2D([0],[0],color='salmon', lw=4),Line2D([0],[0],color='cornflowerblue', lw=4)]

    ax[0].plot(SN_2D.N[5,:], color='purple')
    ax[0].plot(SN_2D.N[25,:], color='blue')
    ax[0].plot(SN_2D.N[50,:], color='red')
    ax[0].plot(SN_2D.N[75,:], color='cyan')
    ax[0].set_xlabel(r'$\mathbf{\Delta}_{ij}$')
    ax[0].set_ylabel(r'$\langle\mathbf{N_{ij}}\rangle$')

    ax[1].plot(SN_2D.N_var[5,:], color='purple')
    ax[1].plot(SN_2D.N_var[25,:], color='blue')
    ax[1].plot(SN_2D.N_var[50,:], color='red')
    ax[1].plot(SN_2D.N_var[75,:], color='cyan')
    ax[1].set_ylabel(r'$Var\;(\mathbf{N_{ij}})$')
    ax[1].set_xlabel(r'$\mathbf{\Delta}_{ij}$')

    plt.tight_layout()
