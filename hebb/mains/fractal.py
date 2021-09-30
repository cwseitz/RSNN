import matplotlib.pyplot as plt
import networkx as nx
from hebb.models import *

mx_lvl = 6
E = 3
sz_cl = 5

f = FractalNetwork(mx_lvl, E, sz_cl)
f.run_generator()
f.plot()
plt.show()
