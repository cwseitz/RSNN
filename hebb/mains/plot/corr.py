import numpy as np
import matplotlib.pyplot as plt

save_dir = '/home/cwseitz/Desktop/data/'
npzfile = 'EIF_LA.npz'
npzfile = np.load(save_dir + npzfile)

At = npzfile['arr_0']
Ft = npzfile['arr_1']
Ct0 = npzfile['arr_2']
f0 = npzfile['arr_3']

plt.imshow(np.real(f0))
plt.show()
