import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from hebb.util import *
from hebb.models import *
from hebb.config import params

save_dir = '/home/cwseitz/Desktop/experiment/'

FLAGS = tf.app.flags.FLAGS

tensor_dict = stack_tensors(save_dir)

fig, ax = plt.subplots()
ax.plot(tensor_dict['sl_1'], color='black', label='Rate Regularization')
plt.show()
