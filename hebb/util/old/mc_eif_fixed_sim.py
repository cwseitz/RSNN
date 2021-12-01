import json
import numpy as np
import matplotlib.pyplot as plt
from hebb.util import *

##################################################
## Run a Monte Carlo simulation for
## excitatory-inhibitory fixed network
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

########################
## Run Monte-Carlo sim
########################

v0min=params['vre'][1]
v0max=params['vT'][1]
v0 = np.random.uniform(v0min, v0max, size=(params['N'],))
v0 = list(v0)
v0 = [x.item() for x in v0]

# print(f'\nThis list should be decreasing for a balanced state to exist: {params['mxe0']/params['mxi0']},{params['wei0']/params['wii0']},{params['wee0']/params['wie0']}\n')
# print(f'Also, this number should be greater than 1: {params['wii0']/params['wee0']}\n')
# print(f'E Rate: {(params['mxe0']*params['wii0']-params['mxi0']*params['wei0')/(params['wei0'*params['wie0']-params['wee0']*params['wii0'])} \n ')
# print(f'I Rate {(params['mxe0']*params['wii0']-params['mxi0']*params['wei0')/(params['wei0'*params['wie0']-params['wee0']*params['wii0'])}')

rnn = ExInEIF_Fixed(params['N'],
                  params['trials'],
                  params['Nrecord'],
                  params['T'],
                  params['Nt'],
                  params['N_e'],
                  params['N_i'],
                  params['q'],
                  params['dt'],
                  params['pee0'],
                  params['pei0'],
                  params['pie0'],
                  params['pii0'],
                  params['jee'],
                  params['jei'],
                  params['jie'],
                  params['jii'],
                  params['wee0'],
                  params['wei0'],
                  params['wie0'],
                  params['wii0'],
                  params['Kee'],
                  params['Kei'],
                  params['Kie'],
                  params['Kii'],
                  params['taux'],
                  params['tausyne'],
                  params['tausyni'],
                  params['tausynx'],
                  params['Jee'],
                  params['Jei'],
                  params['Jie'],
                  params['Jii'],
                  params['maxns'],
                  params['gl'],
                  params['Cm'],
                  params['vlb'],
                  params['vth'],
                  params['DeltaT'],
                  params['vT'],
                  params['vl'],
                  params['vre'],
                  params['tref'],
                  params['mxe'],
                  params['mxi'],
                  params['vxe'],
                  params['vxi'])

rnn.call(v0)

########################
## Extract steady state
########################

t_ss = 1000
spikes = rnn.spikes[:,:,-t_ss:]
ie = rnn.I_e[:,:,-t_ss:]
ii = rnn.I_i[:,:,-t_ss:]
ffwd = rnn.ffwd[:,:,-t_ss:]
v = rnn.V[:,:,-t_ss:]

########################
## Compute cross-spectra
########################

np.savez_compressed(save_dir + 'mc_eif_fixed_v', v)
np.savez_compressed(save_dir + 'mc_eif_fixed_ie', ie)
np.savez_compressed(save_dir + 'mc_eif_fixed_ii', ii)
np.savez_compressed(save_dir + 'mc_eif_fixed_ffwd', ffwd)
np.savez_compressed(save_dir + 'mc_eif_fixed_spikes', spikes)
del rnn
# print('Computing cross spectra...')
# spec = block_spectra(spikes)
# np.savez_compressed(save_dir + 'mc_eif_fixed_spec', spec)
