import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rd
import tensorflow as tf
import params
from branching import *
from time import time
from models import *
from util import *
from conn import *
from plot import *

#Params
FLAGS = tf.app.flags.FLAGS
dt = 1  # time step in ms
tau_m = tau_m_readout = 30
input_f0 = FLAGS.f0 / 1000
thr = FLAGS.thr

#Load model
tensor_dict = stack_tensors()
iter = 0

in_weights = InputWeights(FLAGS.n_in, FLAGS.n_rec)
rec_weights = RecurrentWeights(FLAGS.n_rec, zero=False, pvec=[0.2, 0.3, 0.35, 0.25],
                                ex_mu=-4, ex_sigma=1)

# Load weights from train
# in_weights.conn = tensor_dict['in_conn'][iter]
# in_weights.weights = tensor_dict['in_weights'][iter]
#
# rec_weights.conn = tensor_dict['rec_conn'][iter]
# rec_weights.weights = tensor_dict['rec_weights'][iter]

#Load

# l1 = tensor_dict['l1']
# l2 = tensor_dict['l2']
# rates = tensor_dict['av']
# branching = tensor_dict['branching']


#Cell model

cell = LightLIF(n_in=FLAGS.n_in, n_rec=FLAGS.n_rec, in_weights=in_weights,
                rec_weights=rec_weights,tau=tau_m, thr=thr,dt=dt,
                dampening_factor=FLAGS.dampening_factor,stop_z_gradients=FLAGS.stop_z_gradients)

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
    'input': input,
}

#Unpack results
results_values = sess.run(results_tensors)
input = results_values['input']
spikes = results_values['z']

#Plot
weight_plot(in_weights, rec_weights)
spike_plot(input, spikes)
# train_plot(l1, l2, rates[-1], branching)
plt.show()
