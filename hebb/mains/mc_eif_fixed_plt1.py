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

vfile = 'mc_eif_fixed_v.npz'
iefile = 'mc_eif_fixed_ie.npz'
iifile = 'mc_eif_fixed_ii.npz'
ffwdfile = 'mc_eif_fixed_ffwd.npz'
spikesfile = 'mc_eif_fixed_spikes.npz'

dt = params['dt']
v = np.load(save_dir + vfile)['arr_0']
spikes = np.load(save_dir + spikesfile)['arr_0']
ie = np.load(save_dir + iefile)['arr_0']
ii = np.load(save_dir + iifile)['arr_0']
ffwd = np.load(save_dir + ffwdfile)['arr_0']

#######################################
## Generate a figure summarizing dynamics
#######################################

fig_7(v, ie, ii, ffwd, spikes, dt)
plt.show()
