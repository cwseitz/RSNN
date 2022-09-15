import numpy as np
import matplotlib.pyplot as plt
import json
from hebb.util import *

##################################################
##  Compare theoretical and monte-carlo cross corrs
##################################################
## Author: Clayton Seitz
## Copyright: 2021, The Hebb Project
## Email: cwseitz@uchicago.edu
##################################################

save_dir = '/home/cwseitz/Desktop/data/'

#######################################
## Load the parameters used
#######################################

with open(save_dir + 'params.json', 'r') as fh:
    params = json.load(fh)

#######################################
## Load the Monte-Carlo sim data
#######################################

mc_cc_file = 'mc_eif_rand_c.npz'
mc_cc = np.load(save_dir + mc_cc_file)['arr_0']

# mc_cct_file = 'mc_eif_randt_c.npz'
# mc_cct = np.abs(np.load(save_dir + mc_cc_file)['arr_0'][:,:,0,:])

#######################################
## Load the linear response prediction
#######################################

# lr_cc_file = 'lr_eif_rand_c.npz'
# lr_cc = np.load(save_dir + lr_cc_file)['arr_0']
# lr_cct_file = 'lr_eif_rand_ct.npz'
# lr_cct = np.abs(np.load(save_dir + lr_cct_file)['arr_0'])

# idx_x, idx_y = np.where(~np.eye(lr_cc.shape[0],dtype=bool))
# lr_cc = lr_cc[idx_x,idx_y,:]
# mc_cc = mc_cc[idx_x,idx_y,:]
# lr_cct = lr_cct[idx_x,idx_y,:]
# mc_cct = mc_cct[idx_x,idx_y,:]

plt.imshow(mc_cc[:,:,0])
plt.show()

# plt.scatter(np.abs(lr_cct[:,0]), np.abs(mc_cct[:,0]))
# plt.show()
