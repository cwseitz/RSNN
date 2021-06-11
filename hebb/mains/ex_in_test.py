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

#Cell model
cell = ExInLIF(n_in=FLAGS.n_in, n_rec=FLAGS.n_rec,
                tau=tau_m, thr=thr,dt=dt, p_e=FLAGS.p_e,
                dampening_factor=FLAGS.dampening_factor,
                stop_z_gradients=FLAGS.stop_z_gradients)

cell.w_ee = tensor_dict['w_ee'][iter]
cell.w_ei = tensor_dict['w_ei'][iter]
cell.w_ie = tensor_dict['w_ie'][iter]
cell.w_ii = tensor_dict['w_ii'][iter]
cell.w_e_in = tensor_dict['w_e_in'][iter]
w_ee_grad = tensor_dict['w_ee_grad'][iter]
w_ei_grad = tensor_dict['w_ei_grad'][iter]
w_ie_grad = tensor_dict['w_ie_grad'][iter]
w_ii_grad = tensor_dict['w_ii_grad'][iter]
w_e_in_grad = tensor_dict['w_e_in_grad'][iter]

#Generate test stimulus
frozen_poisson_noise_input = np.random.rand(FLAGS.n_batch, FLAGS.seq_len, FLAGS.n_in) < dt * input_f0
input = tf.constant(frozen_poisson_noise_input, dtype=tf.float32) #only to excitatory units


#Tensorflow ops that simulates the RNN
outs, final_state = tf.nn.dynamic_rnn(cell, input, dtype=tf.float32)
z_e, z_i, v_e, v_i = outs

sess = tf.Session()
sess.run(tf.global_variables_initializer())

results_tensors = {
    'z_e': z_e,
    'z_i': z_i,
    'v_e': v_e,
    'v_i': v_i,
    'input': input
}


#Unpack results
results_values = sess.run(results_tensors)
z_e = results_values['z_e']
z_i = results_values['z_i']
v_e = results_values['v_e']
v_i = results_values['v_i']
input = results_values['input']

#Plot
weight_plot(cell.w_ee, cell.w_ei, cell.w_ie, cell.w_ii, cell.w_e_in)
weight_plot(w_ee_grad, w_ei_grad, w_ie_grad, w_ii_grad, w_e_in_grad)
spike_plot(input[iter], z_e[iter], v_e[iter])
plt.show()
