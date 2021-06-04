import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rd
import tensorflow as tf
import params
from branching import branch_param
from time import time
from models import *
from util import *
from plot import *

FLAGS = tf.app.flags.FLAGS

# Experiment parameters
dt = 1  # time step in ms
input_f0 = FLAGS.f0 / 1000  # input firing rate in kHz in coherence with the usage of ms for time
regularization_f0 = FLAGS.reg_rate / 1000  # desired average firing rate in kHz
tau_m = tau_m_readout = 30
thr = FLAGS.thr

#Cell model
cell = LightLIF(n_in=FLAGS.n_in, n_rec=FLAGS.n_rec, tau=tau_m, thr=thr, dt=dt,
                dampening_factor=FLAGS.dampening_factor,
                stop_z_gradients=FLAGS.stop_z_gradients)


#Stimulus
# mean = np.zeros(FLAGS.n_rec)
# cov = 100*np.eye(FLAGS.n_rec,FLAGS.n_rec)
# input = np.random.multivariate_normal(mean, cov, size=(FLAGS.n_batch, FLAGS.seq_len))
# input = tf.constant(input, dtype=tf.float32)

frozen_poisson_noise_input = np.random.rand(FLAGS.n_batch, FLAGS.seq_len, FLAGS.n_in) < dt * input_f0
input = tf.constant(frozen_poisson_noise_input, dtype=tf.float32)

# Tensorflow ops that simulates the RNN
outs, final_state = tf.nn.dynamic_rnn(cell, input, dtype=tf.float32)
z, v = outs

with tf.name_scope('RegularizationLoss'):

    branching = branch_param(z)
    rec_weights = cell.w_rec_val
    in_weights = cell.w_in_val
    av = tf.reduce_mean(z, axis=(0, 1)) / dt
    num_spikes = tf.reduce_mean(z, axis=(0, 2))/dt
    average_firing_rate_error = av - regularization_f0
    branching_error = num_spikes-regularization_f0

    l1 = 0.5 * tf.reduce_sum(average_firing_rate_error ** 2)
    l2 = 0.5 * tf.reduce_sum(branching_error ** 2)

    overall_loss = l1 + l2

v_scaled = (v - thr) / thr # voltage scaled to be 0 at threshold and -1 at rest
post_term = pseudo_derivative(v_scaled, FLAGS.dampening_factor) / thr # non-linear function of the voltage
z_previous_time = shift_by_one_time_step(z) # z(t-1) instead of z(t)

# put the resulting gradients into lists
var_list = [cell.w_in_var, cell.w_rec_var]
true_gradients = tf.gradients(overall_loss, var_list)

#Optimizer
with tf.name_scope("Optimization"):
    opt = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)
    grads_and_vars = [(g, v) for g, v in zip(true_gradients, var_list)]
    train_step = opt.apply_gradients(grads_and_vars)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

results_tensors = {
    'z': z,
    'av': av,
    'l1': l1,
    'l2': l2,
    'rec_weights': rec_weights,
    'in_weights': in_weights,
    'input': input,
    'num_spikes': num_spikes,
    'branching': branching,
}


def train():

    t_train = 0
    clean_data_dir()

    for k_iter in range(FLAGS.n_iter):
        t0 = time()
        sess.run(train_step)
        t_train = time() - t0
        print(f'Training iteration: {k_iter}')

        if np.mod(k_iter, FLAGS.print_every) == 0:
            t0 = time()
            results_values = sess.run(results_tensors)
            save_tensors(results_values, str(k_iter))
            t_valid = time() - t0

# train()
tensor_dict = stack_tensors()

# fit_power(tensor_dict['z'])

plot1(tensor_dict['z'],
      tensor_dict['l1'],
      tensor_dict['l2'],
      tensor_dict['av'],
      tensor_dict['branching'],
      tensor_dict['input'])
