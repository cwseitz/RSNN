import json
import numpy as np
import matplotlib.pyplot as plt
from rsnn.util import *

##################################################
## Run a Monte Carlo simulation for
## an ensemble of EIF neurons stimulated by GWN
##################################################
## Author: Clayton Seitz
## Copyright: 2021, The Hebb Project
## Email: cwseitz@uchicago.edu
##################################################

save_dir = '/home/cwseitz/Desktop/data/'

########################
## Load Params
########################

with open(save_dir + 'params.json', 'r') as fh:
    params = json.load(fh)

#Load synaptic weights
net = np.load(save_dir + 'mc_eif_ucpld_weights.npz')['arr_0']

ffwd = np.random.normal(params['mxe'],np.sqrt(params['vxe']),size=(params['N'],params['Nt'])).astype(np.float32)

########################
## Run Monte-Carlo sim
########################

v0min=params['vre'][1]
v0max=params['vT'][1]
v0 = np.random.uniform(v0min, v0max, size=(params['N'],))
v0 = list(v0)
v0 = [x.item() for x in v0]

rnn = ExInEIF_Dec(params['N'],
                  params['trials'],
                  params['Nrecord'],
                  params['T'],
                  params['Nt'],
                  params['N_e'],
                  params['N_i'],
                  params['q'],
                  params['dt'],
                  params['taux'],
                  params['tausyne'],
                  params['tausyni'],
                  params['tausynx'],
                  params['maxns'],
                  params['gl'],
                  params['Cm'],
                  params['vlb'],
                  params['vth'],
                  params['DeltaT'],
                  params['vT'],
                  params['vl'],
                  params['vre'],
                  params['tref'])

rnn.call(v0,net,ffwd)

########################
## Extract steady state
########################

t_ss = int(round(params['Tmax']/params['dt']))
t_ss = 1000
spikes = rnn.spikes[:,:,-t_ss:]
ie = rnn.I_e[:,:,-t_ss:]
ii = rnn.I_i[:,:,-t_ss:]
v = rnn.V[:,:,-t_ss:]

########################
## Save data
########################

np.savez_compressed(save_dir + 'mc_eif_ucpld_v', v)
np.savez_compressed(save_dir + 'mc_eif_ucpld_ie', ie)
np.savez_compressed(save_dir + 'mc_eif_ucpld_ii', ii)
np.savez_compressed(save_dir + 'mc_eif_ucpld_ffwd', ffwd)
np.savez_compressed(save_dir + 'mc_eif_ucpld_spikes', spikes)
del rnn
