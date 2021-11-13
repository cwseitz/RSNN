import numpy as np
from hebb.util import *

save_dir = '/home/cwseitz/Desktop/data/'
npzfile = 'data0.npz'
npzfile = np.load(save_dir + npzfile)
v = npzfile['arr_0']
i_e = npzfile['arr_1']
i_i = npzfile['arr_2']
ffwd = npzfile['arr_3']
spikes = npzfile['arr_4']

fig_7(v, i_e, i_i, ffwd, spikes)
plt.show()
