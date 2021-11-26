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
## Load the cross spectra from MC sim
#######################################

spike_spec_file = 'mc_eif_fixed_z_spec.npz'
tot_spec_file = 'mc_eif_fixed_i_spec.npz'
ffwd_spec_file = 'mc_eif_fixed_f_spec.npz'
rec_spec_file = 'mc_eif_fixed_r_spec.npz'

spike_spec = np.load(save_dir + spike_spec_file)['arr_0']
tot_spec = np.load(save_dir + tot_spec_file)['arr_0']
ffwd_spec = np.load(save_dir + ffwd_spec_file)['arr_0']
rec_spec = np.load(save_dir + rec_spec_file)['arr_0']

#######################################
## Plot the cross spectra from MC sim
#######################################

print(spike_spec)
# fig_9(spike_spec, tot_spec, ffwd_spec, rec_spec, params['dt'])
# plt.show()

#######################################
## Filter cells with zero variance in v
#######################################

# dt = params['dt']
# std = v.std(axis=-1)
# idx = np.argwhere(std == 0)
# v = np.delete(v,idx,axis=0)
# ie = np.delete(ie,idx,axis=0)
# ii = np.delete(ii,idx,axis=0)
# ffwd = np.delete(ffwd,idx,axis=0)
#
# #filter neurons which never spike - just to prevent divide by zero
# s = np.sum(spikes[:,0,:], axis=-1)
# idx = np.argwhere(s == 0)
# spikes = np.delete(spikes, idx, axis=0)
# # fig_8(spikes, ie, ii, ffwd, dt)
# # # fig_9(ie, ii, ffwd, spikes, dt*1e-4)
# plt.show()
