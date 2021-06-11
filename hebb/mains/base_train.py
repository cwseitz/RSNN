import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rd
import tensorflow as tf
from time import time

from hebb.util import *
from hebb.models import *
from hebb.config import params

FLAGS = tf.app.flags.FLAGS

# Experiment parameters
dt = 1  # time step in ms
input_f0 = FLAGS.f0 / 1000  # input firing rate in kHz in coherence with the usage of ms for time
regularization_f0 = FLAGS.reg_rate / 1000  # desired average firing rate in kHz
tau_m = tau_m_readout = 30
thr = FLAGS.thr
save_dir = '/home/cwseitz/Desktop/experiment/'


n_excite = int(round(FLAGS.n_rec*FLAGS.p_e))
n_inhib = int(round(FLAGS.n_rec - n_excite))

#Cell model
rec_cmg = ExInConnectivityMatrixGenerator(n_excite, n_inhib, FLAGS.p_ee,
                                          FLAGS.p_ei, FLAGS.p_ie, FLAGS.p_ii,
                                          FLAGS.mu, FLAGS.sigma)
rec_cmg.run_generator()

in_cmg = InputConnectivityGenerator(FLAGS.n_in, FLAGS.n_rec)
in_cmg.run_generator()

cell = ExInLIF(n_in=FLAGS.n_in, n_rec=FLAGS.n_rec,
                tau=tau_m, thr=thr,dt=dt, p_e=0.2,
                dampening_factor=FLAGS.dampening_factor,
                stop_z_gradients=FLAGS.stop_z_gradients)

frozen_poisson_noise_input = np.random.rand(FLAGS.n_batch, FLAGS.seq_len, FLAGS.n_in) < dt * input_f0
input = tf.constant(frozen_poisson_noise_input, dtype=tf.float32)

#Tensorflow ops that simulates the RNN
outs, final_state = tf.nn.dynamic_rnn(cell, input, dtype=tf.float32)
z, v = outs

# with tf.name_scope('RegularizationLoss'):
#
#     target_rec_sign = tf.sign(rec_cmg.weights)
#     rec_sign = tf.sign(cell.w_rec_val)
#     rec_diff = tf.cast(tf.equal(target_rec_sign, rec_sign), tf.float32)
#     rec_reg_loss = tf.reduce_mean(rec_diff)
#
#     #in_reg_loss = -tf.minimum(tf.reduce_min(cell.w_in_val),0)
#
#     target_in_sign = tf.sign(in_cmg.weights)
#     in_sign = tf.sign(cell.w_in_val)
#     in_diff = tf.cast(tf.equal(target_in_sign, in_sign), tf.float32)
#     in_reg_loss = tf.reduce_mean(in_diff)

    # in_diff = cell.w_in_val - in_cmg.weights
    # rec_diff = cell.w_rec_val - rec_cmg.weights
    # rec_reg_loss = 100*tf.reduce_mean(rec_diff**2)
    # in_reg_loss = 100*tf.reduce_mean(in_diff**2)

with tf.name_scope('SpikeRegularizationLoss'):

    # branching = branch_param(z)
    # num_spikes = tf.reduce_mean(z, axis=(0, 2))/dt
    # branching_error = num_spikes-regularization_f0
    # spike_loss_2 = 0.5 * tf.reduce_sum(branching_error ** 2)

    av = tf.reduce_sum(z, axis=0)
    average_firing_rate_error = av - regularization_f0
    spike_loss_1 = 100*tf.reduce_mean(average_firing_rate_error ** 2)

overall_loss = spike_loss_1
v_scaled = (v - thr) / thr # voltage scaled to be 0 at threshold and -1 at rest
post_term = pseudo_derivative(v_scaled, FLAGS.dampening_factor) / thr # non-linear function of the voltage
z_previous_time = shift_by_one_time_step(z) # z(t-1) instead of z(t)

#put the resulting gradients into lists
var_list = [cell.w_in_var, cell.w_rec_var]
true_gradients = tf.gradients(overall_loss, var_list)

#Set gradients for unconnected neurons to zero
# in_gradients, rec_gradients = true_gradients
# in_gradients = tf.multiply(in_gradients, in_cmg.conn)
#rec_gradients = tf.multiply(rec_gradients, rec_cmg.conn)

#Optimizer
with tf.name_scope("Optimization"):
    opt = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)
    grads_and_vars = [(g, v) for g, v in zip(true_gradients, var_list)]
    train_step = opt.apply_gradients(grads_and_vars)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

results_tensors = {
    'z': z,
    'v': v,
    'rec_weights': cell.w_rec_val,
    'in_weights': cell.w_in_val,
    'input': input
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
            save_tensors(results_values, str(k_iter), save_dir=save_dir)
            t_valid = time() - t0

train()
