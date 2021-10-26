import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from hebb.util import *
from hebb.models import *

##################################################
## Main script for simulating an Ornstein-Uhlenbeck
## process with stationary Gaussian noise
##################################################
## Author: Clayton Seitz
## Copyright: 2021, The Hebb Project
## Email: cwseitz@uchicago.edu
##################################################

T = 1.0 #sec
dt = 0.001 #sec
tau = 0.1 #sec
sigma = 1

ou = OrnsteinUhlenbeck(T, dt, tau, sigma, x0=-0.5)
ou.forward()
ou.solve()
ou.histogram()

"""
Probability Densities
"""

steps = [10, 50, 999]
fig, ax = plt.subplots()
add_ou_hist(ax, ou, steps)
plt.show()
