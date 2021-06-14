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
input_f0 = 1 / 1000
thr = FLAGS.thr
save_dir = '/home/cwseitz/Desktop/experiment/'

tensor_dict = stack_tensors(save_dir)

#Load trained model
iter = -1

#Cell model
cell = ExLIF(n_in=FLAGS.n_in, n_rec=FLAGS.n_rec,
                tau=tau_m, thr=thr,dt=dt,
                dampening_factor=FLAGS.dampening_factor,
                stop_z_gradients=FLAGS.stop_z_gradients)

cell.w_ee = tensor_dict['w_ee'][iter]
cell.w_e_in = tensor_dict['w_e_in'][iter]
w_ee_grad = tensor_dict['w_ee_grad'][iter]
w_e_in_grad = tensor_dict['w_e_in_grad'][iter]

#Generate test stimulus
frozen_poisson_noise_input = np.random.rand(FLAGS.n_batch, FLAGS.seq_len, FLAGS.n_in) < dt * input_f0
input = tf.constant(frozen_poisson_noise_input, dtype=tf.float32) #only to excitatory units


#Tensorflow ops that simulates the RNN
outs, final_state = tf.nn.dynamic_rnn(cell, input, dtype=tf.float32)
z_e, v_e = outs

sess = tf.Session()
sess.run(tf.global_variables_initializer())

results_tensors = {
    'z_e': z_e,
    'v_e': v_e,
    'input': input
}


#Unpack results
results_values = sess.run(results_tensors)
z_e = results_values['z_e'][iter]
v_e = results_values['v_e'][iter]
input = results_values['input'][iter]

#Plot
ex_weight_plot(cell.w_ee, cell.w_e_in)
ex_weight_plot(w_ee_grad, w_e_in_grad)
spike_plot(input, z_e, v_e)
plt.show()
