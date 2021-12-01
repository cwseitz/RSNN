import numpy as np
import matplotlib.pyplot as plt
import json
from hebb.util import *

save_dir = '/home/cwseitz/Desktop/data/'
#######################################
## Load the parameters used
#######################################

with open(save_dir + 'params.json', 'r') as fh:
    params = json.load(fh)

#######################################
## Load the Monte-Carlo sim data
#######################################

vfile = 'mc_eif_ucpld_v.npz'
iefile = 'mc_eif_ucpld_ie.npz'
iifile = 'mc_eif_ucpld_ii.npz'
ffwdfile = 'mc_eif_ucpld_ffwd.npz'
spikesfile = 'mc_eif_ucpld_spikes.npz'
p0file = 'lr_eif_ucpld_P0.npz'
r0file = 'lr_eif_ucpld_r0.npz'

dt = params['dt']
v = np.load(save_dir + vfile)['arr_0']
spikes = np.load(save_dir + spikesfile)['arr_0']
ie = np.load(save_dir + iefile)['arr_0']
ii = np.load(save_dir + iifile)['arr_0']
ffwd = np.load(save_dir + ffwdfile)['arr_0']
p0 = np.load(save_dir + p0file)['arr_0']
r0 = np.load(save_dir + r0file)['arr_0']
r0 = 1000*np.round(r0,4)

#######################################
## Generate a figure summarizing dynamics
#######################################

fig = plt.figure(figsize=(7,4))
gs = fig.add_gridspec(4,6, wspace=5, hspace=4)
ax0 = fig.add_subplot(gs[:2, :4])
ax1 = fig.add_subplot(gs[2:4, :2])
ax2 = fig.add_subplot(gs[2:4, 2:4])
ax3 = fig.add_subplot(gs[:2, 4:6])
ax4 = fig.add_subplot(gs[2:4, 4:6])


x = 40; y = 70
add_raster(ax0, spikes[:200,0,-2000:], dt, color='black')
add_unit_voltage(ax1, v, dt, unit=x, trial=0, color='red')
add_unit_voltage(ax1, v, dt, unit=y, trial=0, color='purple')
avg_rate = add_rate_hist(ax2,spikes[:,0,:],dt,min=0,max=3,nbins=5)

cov = np.cov(ffwd)
im = ax4.imshow(cov[:50,:50],cmap='coolwarm')

dV = 0.1
vlb = params['vlb'][0]
vth = params['vth'][0]
v = v.flatten()
bins = np.arange(vlb,vth,dV)
ax3.hist(v,bins=bins,density=True,color='gray',label='MC')
ax3.plot(bins,p0,color='blue',linewidth=3, label='FP')
ax3.set_title(f'Predicted Rate: {r0} Hz', fontsize=8)

fig.colorbar(im,fraction=0.046, pad=0.04)
format_ax(ax0,
          xlabel=r'$\mathrm{Time} \;(\mathrm{ms})$',
          ylabel=r'$\mathrm{Neuron}$',
          ax_is_box=True,
          label_fontsize='medium')

format_ax(ax1,
          xlabel=r'$\mathrm{Time} \;(\mathrm{ms})$',
          ylabel=r'$\mathbf{V} [\mathrm{mV}]$',
          ax_is_box=False,
          label_fontsize='medium')

format_ax(ax2,
          xlabel=r'$\mathrm{Firing\; Rate} \; [\mathrm{Hz}]$',
          ylabel=r'$\mathrm{Counts}$',
          ax_is_box=False,
          label_fontsize='medium')
ax2.set_title(f'$\mu_r$={np.round(avg_rate,3)} Hz',fontsize=10)

format_ax(ax3,
          ax_is_box=False,
          xlabel=r'$V$ [mV]',
          ylabel='PDF')
ax3.legend(loc='upper left', fontsize=8)

ax0.text(-0.1, 1.1, 'A', transform=ax0.transAxes, size=12, weight='bold')
ax1.text(-0.1, 1.1, 'C', transform=ax1.transAxes, size=12, weight='bold')
ax2.text(-0.1, 1.1, 'D', transform=ax2.transAxes, size=12, weight='bold')
ax3.text(-0.1, 1.2, 'B', transform=ax3.transAxes, size=12, weight='bold')
ax4.text(-0.1, 1.2, 'E', transform=ax4.transAxes, size=12, weight='bold')

plt.tight_layout()
plt.show()
