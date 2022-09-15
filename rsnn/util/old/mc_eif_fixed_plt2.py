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

rfile = 'mc_eif_fixed_r_cc.npz'
ffwdfile = 'mc_eif_fixed_f_cc.npz'
totfile = 'mc_eif_fixed_i_cc.npz'
spikesfile = 'mc_eif_fixed_z_cc.npz'

spike_cc = np.load(save_dir + spikesfile)['arr_0']
rec_cc = np.load(save_dir + rfile)['arr_0']
tot_cc = np.load(save_dir + totfile)['arr_0']
ffwd_cc = np.load(save_dir + ffwdfile)['arr_0']

#######################################
## Generate a figure summarizing corrs
#######################################

fig, ax = plt.subplots(2,4)

idx_x, idx_y = np.where(np.eye(spike_cc.shape[0],dtype=bool))
spike_ac = spike_cc[idx_x,idx_y,:,:]
rec_ac = rec_cc[idx_x,idx_y,:,:]
tot_ac = tot_cc[idx_x,idx_y,:,:]
ffwd_ac = ffwd_cc[idx_x,idx_y,:,:]

avg_spike_ac = np.mean(spike_ac,axis=(0,1))
avg_rec_ac = np.mean(rec_ac,axis=(0,1))
avg_tot_ac = np.mean(tot_ac,axis=(0,1))
avg_ffwd_ac = np.mean(ffwd_ac,axis=(0,1))

ax[0,0].plot(avg_spike_ac)
ax[0,1].plot(avg_rec_ac)
ax[0,2].plot(avg_tot_ac)
ax[0,3].plot(avg_ffwd_ac)

idx_x, idx_y = np.where(~np.eye(spike_cc.shape[0],dtype=bool))
spike_cc = spike_cc[idx_x,idx_y,:,:]
rec_cc = rec_cc[idx_x,idx_y,:,:]
tot_cc = tot_cc[idx_x,idx_y,:,:]
ffwd_cc = ffwd_cc[idx_x,idx_y,:,:]

avg_spike_cc = np.mean(spike_cc,axis=(0,1))
avg_rec_cc = np.mean(rec_cc,axis=(0,1))
avg_tot_cc = np.mean(tot_cc,axis=(0,1))
avg_ffwd_cc = np.mean(ffwd_cc,axis=(0,1))

ax[1,0].plot(avg_spike_cc)
ax[1,1].plot(avg_rec_cc)
ax[1,2].plot(avg_tot_cc)
ax[1,3].plot(avg_ffwd_cc)

plt.show()
