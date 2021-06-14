import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from hebb.util import *
from hebb.models import *
from hebb.config import params

save_dir = '/home/cwseitz/Desktop/experiment/'

FLAGS = tf.app.flags.FLAGS

tensor_dict = stack_tensors(save_dir)
print(tensor_dict['sl_1'].shape)
print(tensor_dict['sl_2'].shape)

fig, ax = plt.subplots(1,2)
ax[0].plot(tensor_dict['sl_1'], color='black', label='SL1')
ax[1].plot(tensor_dict['sl_2'], color='blue', label='SL2')
plt.show()
