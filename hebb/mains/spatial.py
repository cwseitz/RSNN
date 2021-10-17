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
N = 100
p_e = 0.8
f = GaussianNetwork(N, p_e, sigma_e=2, sigma_i=2, alpha=10)
