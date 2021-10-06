import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from hebb.util import *
from hebb.models import *

##################################################
## Main script for simulating a 2D lattice of neurons
## with spatially dependent recurrent connectivity
##################################################
## Author: Clayton Seitz
## Copyright: 2021, The Hebb Project
## Email: cwseitz@uchicago.edu
##################################################



#Network
N = 1600
p = 0.2
J_xx = [1, 1, -1, -1]
f = SpatialNetwork2D(N, p, J_xx)
f.plot()
plt.show()

# f.plot()
# plt.show()
