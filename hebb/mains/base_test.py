import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rd
import tensorflow as tf
from time import time

from hebb.util import *
from hebb.models import *
from hebb.config import params

#Params
FLAGS = tf.app.flags.FLAGS
dt = 1  # time step in ms
tau_m = tau_m_readout = 30
input_f0 = FLAGS.f0 / 1000
thr = FLAGS.thr
save_dir = '/home/cwseitz/Desktop/experiment/'

n_excite = int(round(FLAGS.n_rec*FLAGS.p_e))
n_inhib = int(round(FLAGS.n_rec - n_excite))

tensor_dict = stack_tensors(save_dir)

#Load trained model
iter = -1

rec_weights = tensor_dict['rec_weights'][iter]
in_weights = tensor_dict['in_weights'][iter]
weights = [in_weights, rec_weights]

#Cell model
cell = LightLIF(n_in=FLAGS.n_in, n_rec=FLAGS.n_rec, tau=tau_m, thr=thr,dt=dt,
                dampening_factor=FLAGS.dampening_factor, weights=weights,
                stop_z_gradients=FLAGS.stop_z_gradients)

#Generate test stimulus
frozen_poisson_noise_input = np.random.rand(FLAGS.n_batch, FLAGS.seq_len, FLAGS.n_in) < dt * input_f0
input = tf.constant(frozen_poisson_noise_input, dtype=tf.float32)

#Tensorflow ops that simulates the RNN
outs, final_state = tf.nn.dynamic_rnn(cell, input, dtype=tf.float32)
z, v = outs

sess = tf.Session()
sess.run(tf.global_variables_initializer())

results_tensors = {
    'z': z,
    'v': v,
    'input': input,
}

#Unpack results
results_values = sess.run(results_tensors)
input = results_values['input']
spikes = results_values['z']
voltage = results_values['v']

#Plot
weight_plot(in_weights, rec_weights)
spike_plot(input, spikes, voltage)
plt.show()
