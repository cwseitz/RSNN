from hebb.util import *
from hebb.models import *

##################################################
## Main script for simulating a single Hodgkin
## Huxley neuron model
##################################################
## Author: Clayton Seitz
## Copyright: 2021, The Hebb Project
## Email: cwseitz@uchicago.edu
##################################################

dt = 0.1
batches = 1
t = np.arange(0.0, 450.0, dt)
input = 10*(t>100) - 10*(t>200) + 35*(t>300) - 35*(t>400)

hh = HodgkinHuxley(t, input, batches=batches)
hh.call()
hh.plot()
