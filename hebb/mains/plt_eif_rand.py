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

vfile = 'mc_eif_rand_v.npz'
iefile = 'mc_eif_rand_ie.npz'
iifile = 'mc_eif_rand_ii.npz'
ffwdfile = 'mc_eif_rand_ffwd.npz'
spikesfile = 'mc_eif_rand_spikes.npz'
specfile = 'mc_eif_rand_spec.npz'

v = np.load(save_dir + vfile)['arr_0']
ie = np.load(save_dir + iefile)['arr_0']
ii = np.load(save_dir + iifile)['arr_0']
ffwd = np.load(save_dir + ffwdfile)['arr_0']
spikes = np.load(save_dir + spikesfile)['arr_0']
spec = np.load(save_dir + specfile)['arr_0']

#######################################
## Load the linear response prediction
#######################################


#######################################
## Filter cells with zero variance in v
#######################################

dt = params['dt']
std = v.std(axis=-1, keepdims=True)
idx = np.argwhere(std == 0)[0,0]
v = np.delete(v,idx,axis=0)
ie = np.delete(ie,idx,axis=0)
ii = np.delete(ii,idx,axis=0)
ffwd = np.delete(ffwd,idx,axis=0)

#filter neurons which never spike - just to prevent divide by zero
s = np.sum(spikes[:,0,:], axis=-1)
idx = np.argwhere(s == 0)
spikes = np.delete(spikes, idx, axis=0)

fig_8(spikes, ie, ii, ffwd, dt)
fig_9(ie, ii, ffwd, spikes, dt*1e-4)
plt.show()
