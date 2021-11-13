import numpy as np
import matplotlib.pyplot as plt
from hebb.util import *

#load the data
save_dir = '/home/cwseitz/Desktop/data/'
npzfile = 'data0.npz'
npzfile = np.load(save_dir + npzfile)
v = npzfile['arr_0']
i_e = npzfile['arr_1']
i_i = npzfile['arr_2']
ffwd = npzfile['arr_3']
spikes = npzfile['arr_4']

#slice the data to consider only the steady state
t_ss = 2000
dt = 0.1

v = v[:,:,-t_ss:]
i_e = i_e[:,:,-t_ss:]
i_i = i_i[:,:,-t_ss:]
ffwd = ffwd[:,:,-t_ss:]

#filter entries in v, i_e, i_i, with zero variance in v
std = v.std(axis=-1, keepdims=True)
idx = np.argwhere(std == 0)[0,0]
v = np.delete(v,idx,axis=0)
i_e = np.delete(i_e,idx,axis=0)
i_i = np.delete(i_i,idx,axis=0)
ffwd = np.delete(ffwd,idx,axis=0)

fig_7(v, i_e, i_i, ffwd, spikes, dt)
plt.show()
