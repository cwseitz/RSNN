import numpy as np
from hebb.models import *
from hebb.util import *

##################################################
## Main script for simulating a network of LIF
## neurons with fractal connectivity
##################################################
## Author: Clayton Seitz
## Copyright: 2021, The Hebb Project
## Email: cwseitz@uchicago.edu
##################################################

#Connectivity Matrix
mx_lvl = 8
E = 5
sz_cl = 7
net = FractalNetwork(mx_lvl, E, sz_cl)

fig, ax = plt.subplots()
add_fractal_graph(ax, net)
plt.show()
