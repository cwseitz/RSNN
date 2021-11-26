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

#######################################
## Load the linear response prediction
#######################################

lr_cc_file = 'lr_eif_rand_c.npz'
lr_cc = np.load(save_dir + lr_cc_file)['arr_0']

fig, ax = plt.subplots(1,2)
#######################################
## Plot Monte-Carlo average cross corr
#######################################

avg_mc_cc = np.mean(mc_cc,axis=(0,1,2))
ax[0].plot(avg_mc_cc)

#######################################
## Plot linear response average cross corr
#######################################

avg_lr_cc = np.mean(lr_cc,axis=(0,1))
print(avg_lr_cc)
ax[1].plot(avg_lr_cc)
plt.show()
