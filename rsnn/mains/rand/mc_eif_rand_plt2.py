import numpy as np
import matplotlib.pyplot as plt
import json
from matplotlib.ticker import LinearLocator, FormatStrFormatter
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

vfile = 'mc_eif_rand_v.npz'
iefile = 'mc_eif_rand_ie.npz'
iifile = 'mc_eif_rand_ii.npz'
ffwdfile = 'mc_eif_rand_ffwd.npz'
spikesfile = 'mc_eif_rand_spikes.npz'

dt = params['dt']
v = np.load(save_dir + vfile)['arr_0']
spikes = np.load(save_dir + spikesfile)['arr_0']
ie = np.load(save_dir + iefile)['arr_0']
ii = np.load(save_dir + iifile)['arr_0']
ffwd = np.load(save_dir + ffwdfile)['arr_0'][:100,:1000]
ffwd = np.reshape(ffwd, (100,1,1000))

#######################################
## Filter and normalize the data
#######################################

def normalize(x):
    std = x.std(axis=-1, keepdims=True)
    mu = x.mean(axis=-1, keepdims=True)
    x = (x - mu)/std
    return x

dt = params['dt']
s = np.std(ie, axis=-1)
idx = np.argwhere(s == 0)
ie = normalize(np.delete(ie,idx,axis=0))
ii = normalize(np.delete(ii,idx,axis=0))
ffwd = normalize(np.delete(ffwd,idx,axis=0))
s = np.sum(spikes[:,0,:], axis=-1)
idx = np.argwhere(s == 0)
spikes = normalize(np.delete(spikes,idx,axis=0))


#######################################
## Generate a figure
#######################################

fig = plt.figure(figsize=(6,3))
gs = fig.add_gridspec(2,3)
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[0, 1])
ax5 = fig.add_subplot(gs[1, 1])
ax6 = fig.add_subplot(gs[0, 2])
ax7 = fig.add_subplot(gs[1, 2])

rec = ie + ii
total = rec + ffwd
n = spikes.shape[0]
add_mean_ac(ax0,spikes,dt,color='cyan')
add_mean_cc(ax1,spikes,dt,color='cyan')

add_mean_ac(ax4,rec,dt,color='blue')
add_mean_cc(ax5,rec,dt,color='blue')
add_mean_ac(ax6,total,dt,color='red')
add_mean_cc(ax7,total,dt,color='red')

format_ax(ax0,
          xlabel=r'$\mathrm{Lag} \;\tau \;(\mathrm{ms})$',
          ylabel=r'$Z_{xx}(\tau)$',
          ax_is_box=False,
          label_fontsize='medium')

format_ax(ax1,
          xlabel=r'$\mathrm{Lag} \;\tau \;(\mathrm{ms})$',
          ylabel=r'$Z_{xy}(\tau)$',
          ax_is_box=False,
          label_fontsize='medium')

format_ax(ax4,
          xlabel=r'$\mathrm{Lag} \;\tau \;(\mathrm{ms})$',
          ylabel=r'$\langle R_{xx}(\tau)\rangle$',
          ax_is_box=False,
          label_fontsize='medium')

format_ax(ax5,
          xlabel=r'$\mathrm{Lag} \;\tau \;(\mathrm{ms})$',
          ylabel=r'$\langle R_{xy}(\tau)\rangle$',
          ax_is_box=False,
          label_fontsize='medium')

format_ax(ax6,
          xlabel=r'$\mathrm{Lag} \;\tau \;(\mathrm{ms})$',
          ylabel=r'$\langle I_{xx}(\tau)\rangle$',
          ax_is_box=False,
          label_fontsize='medium')

format_ax(ax7,
          xlabel=r'$\mathrm{Lag} \;\tau \;(\mathrm{ms})$',
          ylabel=r'$\langle I_{xy}(\tau)\rangle$',
          ax_is_box=False,
          label_fontsize='medium')

ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.tight_layout()
plt.show()
