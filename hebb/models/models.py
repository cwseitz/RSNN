# # Copyright 2020, the e-prop team
# # Full paper: A solution to the learning dilemma for recurrent networks of spiking neurons
# # Authors: G Bellec*, F Scherr*, A Subramoney, E Hajek, Darjan Salaj, R Legenstein, W Maass
#
# import tensorflow as tf
# import numpy as np
# from collections import namedtuple
#
# Cell = tf.compat.v1.nn.rnn_cell.BasicRNNCell
# ExInLIFStateTuple = namedtuple('ExInLIFStateTuple', ('v_e', 'v_i', 'z_e', 'z_i'))
# ExLIFStateTuple = namedtuple('ExLIFStateTuple', ('v_e','z_e'))
#
# def pseudo_derivative(v_scaled, dampening_factor):
#     '''
#     Define the pseudo derivative used to derive through spikes.
#     :param v_scaled: scaled version of the voltage being 0 at threshold and -1 at rest
#     :param dampening_factor: parameter that stabilizes learning
#     :return:
#     '''
#     return tf.maximum(1 - tf.abs(v_scaled), 0) * dampening_factor
#
#
# @tf.custom_gradient
# def SpikeFunction(v_scaled, dampening_factor):
#     '''
#     The tensorflow function which is defined as a Heaviside function (to compute the spikes),
#     but with a gradient defined with the pseudo derivative.
#     :param v_scaled: scaled version of the voltage being -1 at rest and 0 at the threshold
#     :param dampening_factor: parameter to stabilize learning
#     :return: the spike tensor
#     '''
#     z_ = tf.greater(v_scaled, 0.)
#     z_ = tf.cast(z_, dtype=tf.float32)
#
#     def grad(dy):
#         dE_dz = dy
#         dz_dv_scaled = pseudo_derivative(v_scaled,
#     nsteps = int(round(time/dt))dampening_factor)
#         dE_dv_scaled = dE_dz * dz_dv_scaled
#
#         return [dE_dv_scaled,
#                 tf.zeros_like(dampening_factor)]
#
#     return tf.identity(z_, name="SpikeFunction"), grad
#
# class ExInLIF(Cell):
#     def __init__(self, n_in, n_rec, p_e=0.8, tau=20., thr=0.615,
#                  dt=1., dtype=tf.float32, dampening_factor=0.3,
#                  mu=-0.64, sigma=0.5, stop_z_gradients=False):
#
#         '''
#         A tensorflow RNN cell model to simulate Leaky Integrate and Fire (LIF) neurons.
#         '''
#
#         self.dampening_factor = dampening_factor
#         self.dt = dt
#         self.n_in = n_in
#         self.n_rec = n_rec
#         self.n_excite = int(round(p_e*n_rec))
#         self.n_inhib = self.n_rec - self.n_excite
#         self.data_type = dtype
#         self.stop_z_gradients = stop_z_gradients
#
#         self._num_units = self.n_rec
#
#         self.tau = tf.constant(tau, dtype=dtype)
#         self._decay = tf.exp(-dt / self.tau)
#         self.thr = thr
#
#         with tf.compat.v1.variable_scope('InputWeights'):
#             self.w_e_in_var = tf.Variable(np.random.lognormal(mu, sigma, size=(self.n_in, self.n_excite)), dtype=dtype)
#             self.w_e_in = tf.identity(self.w_e_in_var)
#
#         with tf.compat.v1.variable_scope('RecWeights'):
#
#             self.w_ee_var = tf.Variable(np.random.lognormal(mu, sigma, size=(self.n_excite, self.n_excite)), dtype=dtype)
#             self.w_ei_var = tf.Variable(np.random.lognormal(mu, sigma, size=(self.n_excite, self.n_inhib)), dtype=dtype)
#             self.w_ie_var = tf.Variable(5*np.random.lognormal(mu, sigma, size=(self.n_inhib, self.n_excite)), dtype=dtype)
#             self.w_ii_var = tf.Variable(5*np.random.lognormal(mu, sigma, size=(self.n_inhib, self.n_inhib)), dtype=dtype)
#
#             self.w_ee_disconnect_mask = np.diag(np.ones(self.n_excite, dtype=bool))
#             self.w_ii_disconnect_mask = np.diag(np.ones(self.n_inhib, dtype=bool))
#
#             #Disconnect autotapse
#             self.w_ee = tf.where(self.w_ee_disconnect_mask, tf.zeros_like(self.w_ee_var), self.w_ee_var)
#             self.w_ii = tf.where(self.w_ii_disconnect_mask, tf.zeros_like(self.w_ii_var), self.w_ii_var)
#             self.w_ei = tf.identity(self.w_ei_var)
#             self.w_ie = tf.identity(self.w_ie_var)
#
#
#     @property
#     def state_size(self):
#         return ExInLIFStateTuple(v_e=self.n_excite, v_i=self.n_inhib,
#                                   z_e=self.n_excite, z_i=self.n_inhib)
#
#     @property
#     def output_size(self):
#         return [self.n_excite, self.n_inhib, self.n_excite, self.n_inhib]
#
#     def zero_state(self, batch_size, dtype):
#
#         v_e0 = tf.zeros(shape=(batch_size, self.n_excite), dtype=dtype)
#         v_i0 = tf.zeros(shape=(batch_size, self.n_inhib), dtype=dtype)
#         z_e0 = tf.zeros(shape=(batch_size, self.n_excite), dtype=dtype)
#         z_i0 = tf.zeros(shape=(batch_size, self.n_inhib), dtype=dtype)
#
#         return ExInLIFStateTuple(v_e=v_e0, v_i=v_i0, z_e=z_e0, z_i=z_i0)
#
#     def __call__(self, inputs, state, scope=None, dtype=tf.float32):
#         thr = self.thr
#         z_e = state.z_e
#         v_e = state.v_e
#         z_i = state.z_i
#         v_i = state.v_i
#         decay = self._decay
#
#         #Voltage update - assume everything is a row vector
#         i_e = tf.matmul(inputs, self.w_e_in) + tf.matmul(z_e, self.w_ee) - tf.matmul(z_i, self.w_ie)
#         i_i = tf.matmul(z_e, self.w_ei) - tf.matmul(z_i, self.w_ii)
#
#         i_e_reset = z_e * self.thr * self.dt
#         i_i_reset = z_i * self.thr * self.dt
#         new_v_e = decay * v_e + (1 - decay) * i_e - i_e_reset
#         new_v_i = decay * v_i + (1 - decay) * i_i - i_i_reset
#
#         #Spike generation
#         v_e_scaled = (new_v_e - thr) / thr
#         v_i_scaled = (new_v_i - thr) / thr
#
#         new_z_e = SpikeFunction(v_e_scaled, self.dampening_factor)
#         new_z_i = SpikeFunction(v_i_scaled, self.dampening_factor)
#
#         new_z_e = new_z_e * 1 / self.dt
#         new_z_i = new_z_i * 1 / self.dt
#
#         new_state = ExInLIFStateTuple(v_e=new_v_e, v_i=new_v_i, z_e=new_z_e, z_i=new_z_i)
#
#         return [new_v_e, new_v_i, new_z_e, new_z_i], new_state
#
#
# class ExLIF(Cell):
#     def __init__(self, n_in, n_rec, tau=20., thr=0.615,
#                  dt=1., dtype=tf.float32, dampening_factor=0.3,
#                  mu=-0.64, sigma=0.5, stop_z_gradients=False):
#
#         '''
#         A tensorflow RNN cell model to simulate Leaky Integrate and Fire (LIF) neurons.
#         '''
#
#         self.dampening_factor = dampening_factor
#         self.dt = dt
#         self.n_in = n_in
#         self.n_rec = n_rec
#         self.data_type = dtype
#         self.stop_z_gradients = stop_z_gradients
#
#         self._num_units = self.n_rec
#
#         self.tau = tf.constant(tau, dtype=dtype)
#         self._decay = tf.exp(-dt / self.tau)
#         self.thr = thr
#
#         with tf.variable_scope('InputWeights'):
#             self.w_e_in_var = tf.Variable(np.random.randn(n_in, n_rec) / np.sqrt(n_in), dtype=dtype)
#             self.w_e_in = tf.identity(self.w_e_in_var)
#
#         with tf.variable_scope('RecWeights'):
#             self.w_ee_var = tf.Variable(np.random.randn(n_rec, n_rec) / np.sqrt(n_rec), dtype=dtype)
#             self.recurrent_disconnect_mask = np.diag(np.ones(n_rec, dtype=bool))
#             self.w_ee = tf.where(self.recurrent_disconnect_mask, tf.zeros_like(self.w_ee_var),
#                                       self.w_ee_var)  # Disconnect autotapse
#
#         # with tf.variable_scope('InputWeights'):
#         #     self.w_e_in = tf.Variable(np.random.lognormal(mu, sigma, size=(self.n_in, self.n_rec)), dtype=dtype)
#         #
#         # with tf.variable_scope('RecWeights'):
#         #
#         #     self.w_ee = tf.Variable(np.random.lognormal(mu, sigma, size=(self.n_rec, self.n_rec)), dtype=dtype)
#         #     self.w_ee_disconnect_mask = np.diag(np.ones(self.n_rec, dtype=bool))
#         #
#         #     #Disconnect autotapse
#         #     self.w_ee = tf.Variable(tf.where(self.w_ee_disconnect_mask, tf.zeros_like(self.w_ee), self.w_ee))
#
#
#     @property
#     def state_size(self):
#         return ExLIFStateTuple(v_e=self.n_rec, z_e=self.n_rec)
#
#     @property
#     def output_size(self):
#         return [self.n_rec, self.n_rec]
#
#     def zero_state(self, batch_size, dtype):
#
#         v_e0 = tf.zeros(shape=(batch_size, self.n_rec), dtype=dtype)
#         z_e0 = tf.zeros(shape=(batch_size, self.n_rec), dtype=dtype)
#
#         return ExLIFStateTuple(v_e=v_e0, z_e=z_e0)
#
#     def __call__(self, inputs, state, scope=None, dtype=tf.float32):
#         thr = self.thr
#         z_e = state.z_e
#         v_e = state.v_e
#         decay = self._decay
#
#         # if self.stop_z_gradients:
#         #     z_e = tf.stop_gradient(z_e)
#         #     z_i = tf.stop_gradient(z_i)
#
#         #Voltage update - assume everything is a row vector
#         i_e = tf.matmul(inputs, self.w_e_in) + tf.matmul(z_e, self.w_ee)
#         i_e_reset = z_e * self.thr * self.dt
#         new_v_e = decay * v_e + (1 - decay) * i_e - i_e_reset
#
#         #Spike generation
#         v_e_scaled = (new_v_e - thr) / thr
#
#         new_z_e = SpikeFunction(v_e_scaled, self.dampening_factor)
#         new_z_e = new_z_e * 1 / self.dt
#
#         new_state = ExLIFStateTuple(v_e=new_v_e, z_e=new_z_e)
#
#         return [new_z_e, new_v_e], new_state
#
# # class LightLIF(Cell):
# #     def __init__(self, n_in, n_rec, tau=20., thr=0.615,
# #                  dt=1., dtype=tf.float32, dampening_factor=0.3,
# #                  weights=None, stop_z_gradients=False):
# #
# #         '''
# #         A tensorflow RNN cell model to simulate Leaky Integrate and Fire (LIF) neurons.
# #
# #         WARNING: This model might not be compatible with tensorflow framework extensions because the input and recurrent
# #         weights are defined with tf.Variable at creation of the cell instead of using variable scopes.
# #
# #         :param n_in: number of input neurons
# #         :param n_rec: number of recurrent neurons
# #         :param tau: membrane time constant
# #         :param thr: threshold voltage
# #         :param dt: time step
# #         :param dtype: data type
# #         :param dampening_factor: parameter to stabilize learning
# #         :param stop_z_gradients: if true, some gradients are stopped to get an equivalence between eprop and bptt
# #         '''
# #
# #         self.dampening_factor = dampening_factor
# #         self.dt = dt
# #         self.n_in = n_in
# #         self.n_rec = n_rec
# #         self.data_type = dtype
# #         self.stop_z_gradients = stop_z_gradients
# #
# #         self._num_units = self.n_rec
# #
# #         self.tau = tf.constant(tau, dtype=dtype)
# #         self._decay = tf.exp(-dt / self.tau)
# #         self.thr = thr
# #
# #         if weights is not None:
# #
# #             in_weights, rec_weights = weights
# #             with tf.variable_scope('InputWeights'):
# #                 self.w_in_var = tf.Variable(in_weights, dtype=dtype)
# #                 self.w_in_val = tf.identity(self.w_in_var)
# #
# #             with tf.variable_scope('RecWeights'):
# #                 self.w_rec_var = tf.Variable(rec_weights, dtype=dtype)
# #                 self.w_rec_val = tf.identity(self.w_rec_var)
# #
# #         else:
# #
# #             with tf.variable_scope('InputWeights'):
# #                 self.w_in_var = tf.Variable(np.random.randn(n_in, n_rec) / np.sqrt(n_in), dtype=dtype)
# #                 self.w_in_val = tf.identity(self.w_in_var)
# #
# #             with tf.variable_scope('RecWeights'):
# #                 self.w_rec_var = tf.Variable(np.random.randn(n_rec, n_rec) / np.sqrt(n_rec), dtype=dtype)
# #                 self.recurrent_disconnect_mask = np.diag(np.ones(n_rec, dtype=bool))
# #                 self.w_rec_val = tf.where(self.recurrent_disconnect_mask, tf.zeros_like(self.w_rec_var),
# #                                           self.w_rec_var)  # Disconnect autotapse
# #
# #
# #     @property
# #     def state_size(self):
# #         return LightLIFStateTuple(v=self.n_rec, z=self.n_rec)
# #
# #     @property
# #     def output_size(self):
# #         return [self.n_rec, self.n_rec]
# #
# #     def zero_state(self, batch_size, dtype, n_rec=None):
# #         if n_rec is None: n_rec = self.n_rec
# #
# #         v0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype)
# #         z0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype)
# #
# #         return LightLIFStateTuple(v=v0, z=z0)
# #
# #     def __call__(self, inputs, state, scope=None, dtype=tf.float32):
# #         thr = self.thr
# #         z = state.z
# #         v = state.v
# #         decay = self._decay
# #
# #         if self.stop_z_gradients:
# #             z = tf.stop_gradient(z)
# #
# #         # update the voltage
# #         i_t = tf.matmul(inputs, self.w_in_val) + tf.matmul(z, tf.transpose(self.w_rec_val))
# #         #i_t = inputs + tf.matmul(z, self.w_rec_val)
# #         I_reset = z * self.thr * self.dt
# #         new_v = decay * v + (1 - decay) * i_t - I_reset
# #
# #         # Spike generation
# #         v_scaled = (new_v - thr) / thr
# #         new_z = SpikeFunction(v_scaled, self.dampening_factor)
# #         new_z = new_z * 1 / self.dt
# #         new_state = LightLIFStateTuple(v=new_v, z=new_z)
# #         return [new_z, new_v], new_state
#
#
# def shift_by_one_time_step(tensor, initializer=None):
#     '''
#     Shift the input on the time dimension by one.
#     :param tensor: a tensor of shape (trial, time, neuron)
#     :param initializer: pre-prend this as the new first element on the time dimension
#     :return: a shifted tensor of shape (trial, time, neuron)
#     '''
#     with tf.name_scope('TimeShift'):
#         assert tensor.dtype in [tf.float16, tf.float32, tf.float64]
#         r_shp = range(len(tensor.get_shape()))
#         transpose_perm = [1, 0] + list(r_shp)[2:]
#         tensor_time_major = tf.transpose(tensor, perm=transpose_perm)
#
#         if initializer is None:
#             initializer = tf.zeros_like(tensor_time_major[0])
#
#         shifted_tensor = tf.concat([initializer[None, :, :], tensor_time_major[:-1]], axis=0)
#
#         shifted_tensor = tf.transpose(shifted_tensor, perm=transpose_perm)
#     return shifted_tensor
