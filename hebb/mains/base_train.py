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

cell = LightLIF(n_in=FLAGS.n_in, n_rec=FLAGS.n_rec, in_weights=in_cmg,
                rec_weights=rec_cmg,tau=tau_m, thr=thr,dt=dt,
                dampening_factor=FLAGS.dampening_factor,stop_z_gradients=FLAGS.stop_z_gradients)


frozen_poisson_noise_input = np.random.rand(FLAGS.n_batch, FLAGS.seq_len, FLAGS.n_in) < dt * input_f0
input = tf.constant(frozen_poisson_noise_input, dtype=tf.float32)

#Tensorflow ops that simulates the RNN
outs, final_state = tf.nn.dynamic_rnn(cell, input, dtype=tf.float32)
z, v = outs

with tf.name_scope('RegularizationLoss'):

    #This loss maintains the sign of inhibitory and excitatory units
    wmat = np.ones([FLAGS.n_rec, FLAGS.n_rec])
    wmat[n_excite:,:] = -1 * wmat[n_excite:,:]
    wmat[np.where(rec_cmg.conn == 0)] = 0

    rec_sign = tf.convert_to_tensor(wmat, dtype=tf.float32)
    sign_error = 1-tf.cast(tf.equal(rec_sign, tf.sign(rec_cmg.weights)), tf.float32)
    reg_loss = tf.reduce_mean(sign_error**2)


    # diff = tf.multiply(rec_cmg.conn, rec_cmg.weights)
    # reg_loss = tf.reduce_mean((rec_cmg.weights - diff)**2)

with tf.name_scope('SpikeRegularizationLoss'):

    branching = branch_param(z)
    av = tf.reduce_mean(z, axis=(0, 1)) / dt
    num_spikes = tf.reduce_mean(z, axis=(0, 2))/dt
    average_firing_rate_error = av - regularization_f0
    branching_error = num_spikes-regularization_f0

    spike_loss_1 = 0.5 * tf.reduce_sum(average_firing_rate_error ** 2)
    spike_loss_2 = 0.5 * tf.reduce_sum(branching_error ** 2)

    spike_loss_loss = spike_loss_1 + spike_loss_2

overall_loss = reg_loss + spike_loss_loss
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
    'v': v,
    'av': av,
    'spike_loss_1': spike_loss_1,
    'spike_loss_2': spike_loss_2,
    'reg_loss': reg_loss,
    'rec_weights': cell.w_rec_val,
    'in_weights': cell.w_in_val,
    'rec_conn': cell.w_rec_conn,
    'in_conn': cell.w_in_conn,
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
            save_tensors(results_values, str(k_iter), save_dir=save_dir)
            t_valid = time() - t0

train()
