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

def sim(p_in, p_rec):

    arr = []
    for i in range(5):

        #Rewire
        in_weights = InputWeights(FLAGS.n_in, FLAGS.n_rec, p=p_in)
        rec_weights = RecurrentWeights(FLAGS.n_rec, zero=False, pvec=[p_rec, 0.3, 0.35, 0.25],
                                        ex_mu=-2, ex_sigma=1.2)

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
        arr.append(spikes)
    arr = np.array(arr)
    new_shape = (arr.shape[0]*arr.shape[1], arr.shape[2], arr.shape[3])
    arr = np.reshape(arr, new_shape)

    return np.array(arr)

def compute_entropy(prob):
    prob = prob[np.nonzero(prob)]
    ent = -np.sum(prob*np.log2(prob))
    ent = np.round(ent, 2)
    return ent

p_in = 0.1
p_rec = np.linspace(0.1, 0.9, 5)
prob_arr = []

for p in p_rec:
    m = sim(p_in, p)
    prob = pattern_prob(m)
    prob_arr.append(prob)

fig, ax = plt.subplots(1,2)
colors=['red', 'blue', 'cyan', 'black', 'purple']
entropies = []
for i, prob in enumerate(prob_arr):

    ent = compute_entropy(prob)
    entropies.append(ent)
    ax[0].plot(prob, label=f'p_rec={np.round(p_rec[i],2)}', color=colors[i])

ax[0].set_xlabel('Pattern index')
ax[0].set_ylabel('log P')
ax[0].legend()
ax[1].plot(p_rec, entropies, color='red')
ax[1].set_xlabel('p_rec')
ax[1].set_ylabel('H(P)')
ax[0].set_yscale('log')

plt.legend()
plt.show()
