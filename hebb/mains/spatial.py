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
N = 525
p_e = 0.8
net = GaussianNetwork(N, p_e, sigma_e=1, sigma_i=1, alpha=10)
fig, ax = plt.subplots()
add_spectral_graph(ax, net)
plt.show()
