import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from hebb.util import *
from hebb.models import *
from hebb.config import params

save_dir = '/home/cwseitz/Desktop/experiment/'

FLAGS = tf.app.flags.FLAGS
n_excite = int(round(FLAGS.n_rec*FLAGS.p_e))
n_inhib = int(round(FLAGS.n_rec - n_excite))
tensor_dict = stack_tensors(save_dir)
iter = 2

rec_cmg = ExInConnectivityMatrixGenerator(n_excite, n_inhib, FLAGS.p_ee,
                                          FLAGS.p_ei, FLAGS.p_ie, FLAGS.p_ii,
                                          FLAGS.mu, FLAGS.sigma)

in_cmg = InputConnectivityGenerator(FLAGS.n_in, FLAGS.n_rec)

input = tensor_dict['input'][iter]
spikes = tensor_dict['z'][iter]
voltage = tensor_dict['v'][iter]
spike_reg_1 = tensor_dict['spike_loss_1'][iter]
spike_reg_2 = tensor_dict['spike_loss_2'][iter]
reg_loss = tensor_dict['reg_loss']
rec_cmg.conn = tensor_dict['rec_conn'][iter]
rec_cmg.weights = tensor_dict['rec_weights'][iter]
in_cmg.conn = tensor_dict['in_conn'][iter]
in_cmg.weights = tensor_dict['in_weights'][iter]

print(reg_loss)
weight_plot(in_cmg, rec_cmg)
plt.show()
train_plot(spike_reg_1, spike_reg_2, reg_loss)
