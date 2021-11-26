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

iefile = 'mc_eif_fixed_ie.npz'
iifile = 'mc_eif_fixed_ii.npz'
ffwdfile = 'mc_eif_fixed_ffwd.npz'
spikesfile = 'mc_eif_fixed_spikes.npz'

#######################################
## Compute cross spectra of F,R,I,Z
#######################################

spikes = np.load(save_dir + spikesfile)['arr_0']
print('Computing Z cross spectra...')
spec = block_spectra(spikes)
np.savez_compressed(save_dir + 'mc_eif_fixed_z_spec', spec)
del spec; del spikes

print('Computing R cross spectra...')
ie = np.load(save_dir + iefile)['arr_0']
ii = np.load(save_dir + iifile)['arr_0']
total = ie + ii
spec = block_spectra(total)
np.savez_compressed(save_dir + 'mc_eif_fixed_r_spec', spec)
del spec; del ie; del ii

print('Computing I cross spectra...')
ffwd = np.load(save_dir + ffwdfile)['arr_0']
spec = block_spectra(ffwd+total)
np.savez_compressed(save_dir + 'mc_eif_fixed_i_spec', spec)
del spec; del total


print('Computing F cross spectra...')
spec = block_spectra(ffwd)
np.savez_compressed(save_dir + 'mc_eif_fixed_f_spec', spec)
del spec; del ffwd
