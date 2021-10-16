import numpy as np
import matplotlib.pyplot as plt

N = 1600
M = np.sqrt(N)
p_e = 0.8
ex_in = np.random.uniform(0,1,size=(N,))
ex_in = ex_in < p_e #1 for excitatory, 0 for inhibitory
J = np.zeros((N,N))

idx_x, idx_y = np.triu_indices(self.N) #upper triangle indices

for k in range(len(idx_x)):
    i,j = idx_x[k], idx_y[k]
    m = np.mod(j, M)
    d = distance(i,j)
