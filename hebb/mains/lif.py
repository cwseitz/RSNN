import numpy as np
import scipy as p
import matplotlib.pyplot as plt
from hebb.util import *
from hebb.models import *

dt = 0.1
batches = 2
t = sp.arange(0.0, 50.0, dt)
input = 0.1*(t>10) - 0.1*(t>20) + 0.35*(t>30) - 0.35*(t>40)
input *= 5

lif = LIF(t, batches=batches)
lif.call(input, plot=True)
