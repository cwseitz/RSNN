import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rd
import tensorflow as tf
from time import time

from hebb.util import *
from hebb.models import *
from hebb.config import params

tf.compat.v1.disable_eager_execution()
FLAGS = tf.compat.v1.app.flags.FLAGS

# Experiment parameters
dt = 1  # time step in ms
input_f0 = FLAGS.f0 / 1000  # input firing rate in kHz in coherence with the usage of ms for time
regularization_f0 = 0.02  # desired firing rate in spikes/ms
tau_m = tau_m_readout = 30
thr = FLAGS.thr


cell = ExInLIF(n_in=FLAGS.n_in, n_rec=FLAGS.n_rec,
                tau=tau_m, thr=thr,dt=dt, p_e=FLAGS.p_e,
                dampening_factor=FLAGS.dampening_factor,
                stop_z_gradients=FLAGS.stop_z_gradients)


frozen_poisson_noise_input = np.random.rand(FLAGS.n_batch, FLAGS.seq_len, FLAGS.n_in) < dt * input_f0
input = tf.constant(frozen_poisson_noise_input, dtype=tf.float32) #only to excitatory units

#Tensorflow ops that simulates the RNN
outs, final_state = tf.compat.v1.nn.dynamic_rnn(cell, input, dtype=tf.float32)
v_e, v_i, z_e, z_i = outs

with tf.name_scope('SpikeRegularizationLoss'):

    z = tf.concat([z_e, z_i], axis=-1)
    av_1 = tf.reduce_mean(z, axis=(0, 1))
    firing_rate_error_1 = av_1 - regularization_f0
    sl_1 = 0.5 * tf.reduce_sum(firing_rate_error_1 ** 2)

    av_2 = tf.reduce_mean(z, axis=(0, 2))
    firing_rate_error_2 = av_2 - regularization_f0
    sl_2 = 0.5 * tf.reduce_sum(firing_rate_error_2 ** 2)

#Gradients
alpha, beta = 10, 10
overall_loss = alpha*sl_1 + beta*sl_2
var_list = [cell.w_e_in_var, cell.w_ee_var, cell.w_ei_var, cell.w_ie_var, cell.w_ii_var]

true_gradients = tf.compat.v1.gradients(overall_loss, var_list)
w_e_in_grad, w_ee_grad, w_ei_grad, w_ie_grad, w_ii_grad = true_gradients

#Optimizer
with tf.name_scope("Optimization"):
    opt = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)
    grads_and_vars = [(g, v) for g, v in zip(true_gradients, var_list)]
    train_step = opt.apply_gradients(grads_and_vars)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

results_tensors = {
    'sl_1': sl_1,
    'sl_2': sl_2,
    'z_e': z_e,
    'z_i': z_i,
    'v_e': v_e,
    'v_i': v_i,
    'w_e_in': cell.w_e_in,
    'w_ee': cell.w_ee,
    'w_ei': cell.w_ei,
    'w_ie': cell.w_ie,
    'w_ii': cell.w_ii,
    'w_e_in_grad': w_e_in_grad,
    'w_ee_grad': w_ee_grad,
    'w_ei_grad': w_ei_grad,
    'w_ie_grad': w_ie_grad,
    'w_ii_grad': w_ii_grad,
    'input': input
}


def train():

    t_train = 0

    for k_iter in range(FLAGS.n_iter):
        t0 = time()
        sess.run(train_step)
        t_train = time() - t0
        print(f'Training iteration: {k_iter}')

        if np.mod(k_iter, FLAGS.print_every) == 0:
            t0 = time()
            results_values = sess.run(results_tensors)
            save_tensors(results_values, str(k_iter), save_dir='data/')
            t_valid = time() - t0

train()
