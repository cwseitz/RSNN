import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import cm
from hebb.util import *
from hebb.models import *

def multi_gauss(r, mu, sigma):

    n = mu.shape[0]
    det = np.linalg.det(sigma)
    inv = np.linalg.inv(sigma)
    N = np.sqrt((2*np.pi)**n * det)
    fac = np.einsum('...k,kl,...l->...', r-mu, inv, r-mu)
    return np.exp(-fac / 2) / N

def prod_gauss(mu_1, sigma_1, mu_2, sigma_2):

    mu_3 = sigma_2 @ np.linalg.inv(sigma_1+sigma_2) @ mu_1 +\
           sigma_1 @ np.linalg.inv(sigma_1+sigma_2) @ mu_2

    sigma_3 = sigma_1 @ np.linalg.inv(sigma_1+sigma_2) @ sigma_2

    return mu_3, sigma_3

def func(r, mu_1, sigma_1, mu_2, sigma_2):

    mu_3, sigma_3 = prod_gauss(mu_1, sigma_1, mu_2, sigma_2)
    rho_1 = rho_2 = 0.5
    Z_1 = rho_1*multi_gauss(r, mu_1, sigma_1)
    Z_2 = rho_2*multi_gauss(r, mu_2, sigma_2)
    Z_3 = Z_1*Z_2
    Z = Z_1 + Z_2
    return Z, Z_3

N = 40
X = np.linspace(-2, 2, N)
Y = np.linspace(-2, 2, N)
X, Y = np.meshgrid(X, Y)
r = np.empty(X.shape + (2,))
r[:, :, 0] = X
r[:, :, 1] = Y

N_12 = []
N_12_var = []

seps = np.arange(0.1, 1, 0.05) #range for the distance between neurons
sigs = np.arange(0.1, 1, 0.05) #range for the broadness of synaptic connections

for sig in sigs:
    N_12_sig = []
    N_12_var_sig = []
    for sep in seps:
        mu_1 = np.array([-sep,0])
        sigma_1 = np.array([[sig,0],[0,sig]])
        mu_2 = np.array([sep,0])
        sigma_2 = np.array([[sig,0],[0,sig]])
        Z, Z_3 = func(r, mu_1, sigma_1, mu_2, sigma_2)
        N_12_sig.append(np.sum(Z_3))
        N_12_var_sig.append(np.sum(Z_3*(1-Z_3)))
    N_12.append(N_12_sig)
    N_12_var.append(N_12_var_sig)

N_12 = np.array(N_12)
N_12_std = np.sqrt(np.array(N_12_var))

custom_lines = [Line2D([0],[0],color='salmon', lw=4),Line2D([0],[0],color='cornflowerblue', lw=4)]
fig, ax = plt.subplots(2,2,figsize=(5,5))
ax[0,0].contour(X, Y, Z, cmap=cm.Reds)
ax[0,0].grid(False)
ax[0,0].set_xticks([])
ax[0,0].set_yticks([])
ax[0,0].set_xlabel(r'$x_1$')
ax[0,0].set_ylabel(r'$x_2$')
ax[0,0].contour(X, Y, Z_3, cmap=cm.Blues)
ax[0,0].legend(custom_lines, [r'$f_{\alpha} + f_{\beta}$', r'$f_{\alpha\beta}$'])
ax[0,1].plot(N_12_std[0], color='black', label='$\sigma$')
ax[0,1].plot(N_12[0], color='cyan', label='$\mu$')
ax[0,1].set_xlabel(r'$\mathbf{\Delta}_{\alpha\beta}$')
ax[0,1].legend()
ax[1,0].imshow(N_12, cmap='coolwarm', vmin=0, vmax=N_12.max())
ax[1,0].set_xlabel(r'$\mathbf{\Delta}_{\alpha\beta}$')
ax[1,0].set_ylabel(r'$\mathbf{\sigma}$')
ax[1,1].imshow(N_12_std, cmap='coolwarm', vmin=0, vmax=N_12_std.max())
ax[1,1].set_ylabel(r'$\mathbf{\sigma}$')
ax[1,1].set_xlabel(r'$\mathbf{\Delta}_{\alpha\beta}$')
plt.tight_layout()
plt.show()

# T = 1
# dt = 0.001
# mu = [0.1, 0.1] #one entry per trial (neuron)
# cov = np.array([[0.1,0.05],[0.05,0.1]]) #covariance between trials (different neurons)
#
# ex_in_ou = ExInOU(T, dt, mu, cov)
# ex_in_ou.forward()
#
# fig,ax = plt.subplots(1,3)
# ax[0].scatter(ex_in_ou.eta[0,:], ex_in_ou.eta[1,:], color='red')
# ax[0].set_xlabel('$\mathbf{PSP_1} \; [\mathrm{mV}]$')
# ax[0].set_ylabel('$\mathbf{PSP_2} \; [\mathrm{mV}]$')
# ax[1].plot(ex_in_ou.v[0,:], color='red')
# ax[1].plot(ex_in_ou.v[1,:], color='blue')
#
#
# rho = 1e4
# sigma_1 = 100
# sigma_2 = 100
#
# im_1 = np.zeros((1000,1000)); im_1[49, 400] = rho
# im_1 = gaussian_filter(im_1, sigma_1)
#
# im_2 = np.zeros((1000,1000)); im_2[49, 600] = rho
# im_2 = gaussian_filter(im_2, sigma_2)
#
# prod = im_1*im_2
# ax[2].plot(im_1[49,:], color='red', label=f'$\mu={np.round(np.sum(im_1), 2)}$')
# ax[2].plot(im_2[49,:], color='blue', label=f'$\mu={np.round(np.sum(im_2), 2)}$')
# ax[2].plot(prod[49,:], color='cyan', label=f'$\mu={np.round(np.sum(prod), 2)}$')
# # ax[2].set_xlim([40, 60])
# ax[2].legend()
#
# plt.tight_layout()
# plt.show()
