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
n = 200
dt = 0.1

v = v[:n,:,-t_ss:]
i_e = i_e[:n,:,-t_ss:]
i_i = i_i[:n,:,-t_ss:]
ffwd = ffwd[:n,:,-t_ss:]
spikes = spikes[:n,:,-t_ss:]

#filter entries in v, i_e, i_i, with zero variance in v
std = v.std(axis=-1, keepdims=True)
idx = np.argwhere(std == 0)[0,0]
v = np.delete(v,idx,axis=0)
i_e = np.delete(i_e,idx,axis=0)
i_i = np.delete(i_i,idx,axis=0)
ffwd = np.delete(ffwd,idx,axis=0)

#filter neurons which never spike - just to prevent divide by zero
s = np.sum(spikes[:,0,:], axis=-1)
idx = np.argwhere(s == 0)
spikes = np.delete(spikes, idx, axis=0)

fig_8(spikes, i_e, i_i, ffwd, dt)
fig_9(i_e, i_i, ffwd, spikes, dt*1e-4)
plt.show()
