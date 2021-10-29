import numpy as np
import matplotlib.pyplot as plt
from hebb.models import *

# N = 900; q = 0.8
# M = int(round(np.sqrt(N)))
#
# sigma = np.sqrt(N)/8
# net = GaussianNetwork(N, sigma*np.ones((M,M)), q)
# vals, vecs = np.linalg.eig(net.C)
# unique_vals, unique_counts = np.unique(vals, return_counts=True)
# ortho = vecs.T @ vecs
# rank = np.linalg.matrix_rank(net.C)
# print(rank)
# print(ortho)
# print(vals.shape, unique_vals.shape)
# plt.imshow(np.abs(ortho))
# plt.show()

x = np.random.normal(0,1,size=(1000,))
y = np.random.normal(0,1,size=(1000,))
cc = np.correlate(x,y, mode='full')
fig, ax = plt.subplots(1,2)
ax[0].plot(x)
ax[0].plot(y)
ax[1].plot(cc)
plt.show()
