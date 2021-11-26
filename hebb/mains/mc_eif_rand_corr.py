import numpy as np
import matplotlib.pyplot as plt
import json
from hebb.util import *

def normalize(x):
    std = x.std(axis=-1, keepdims=True)
    mu = x.mean(axis=-1, keepdims=True)
    x = (x - mu)/std
    return x

save_dir = '/home/cwseitz/Desktop/data/'
#######################################
## Load the parameters used
#######################################

with open(save_dir + 'params.json', 'r') as fh:
    params = json.load(fh)

#######################################
## Load the Monte-Carlo sim data
#######################################

iefile = 'mc_eif_rand_ie.npz'
iifile = 'mc_eif_rand_ii.npz'
ffwdfile = 'mc_eif_rand_ffwd.npz'
spikesfile = 'mc_eif_rand_spikes.npz'

spikes = np.load(save_dir + spikesfile)['arr_0']
ie = np.load(save_dir + iefile)['arr_0']
ii = np.load(save_dir + iifile)['arr_0']
ffwd = np.load(save_dir + ffwdfile)['arr_0']

#######################################
## Clean and normalize the data
#######################################

dt = params['dt']
s = np.std(ie, axis=-1)
idx = np.argwhere(s == 0)
ie = normalize(np.delete(ie,idx,axis=0))
ii = normalize(np.delete(ii,idx,axis=0))
ffwd = normalize(np.delete(ffwd,idx,axis=0))
s = np.sum(spikes[:,0,:], axis=-1)
idx = np.argwhere(s == 0)
# spikes = normalize(np.delete(spikes,idx,axis=0))
spikes = np.delete(spikes,idx,axis=0)

#######################################
## Compute cross-corr of F,R,I,Z
#######################################

total = ie + ii + ffwd
rec = ie + ii

print('Computing Z cross-corr...')
spike_cc = block_cc(spikes,params['nfreq'])
np.savez_compressed(save_dir + 'mc_eif_rand_c', spike_cc)

# print('Computing R cross-corr...')
# total_cc = block_cc(total)
# np.savez_compressed(save_dir + 'mc_eif_rand_r_cc', total_cc)
#
# print('Computing I cross-corr...')
# ffwd_cc = block_cc(ffwd)
# np.savez_compressed(save_dir + 'mc_eif_rand_i_cc', ffwd_cc)
#
# print('Computing F cross-corr...')
# rec_cc = block_cc(rec)
# np.savez_compressed(save_dir + 'mc_eif_rand_f_cc', rec_cc)
