import numpy as np
import tensorflow as tf

def activity(spikes):

	trial_idx, seq_len, units = spikes.shape
	a = tf.reduce_sum(spikes, axis=-1)

	return a


def branch_param(spikes, lag=0, tau=1):

	trial_idx, seq_len, units = spikes.shape
	a = tf.reduce_sum(spikes, axis=-1)
	r_arr = []
	for t in range(lag+1, seq_len-tau-lag-1):
		slice = a[:,t+lag:t+lag+tau]
		x = a[:, t]
		s = tf.reduce_sum(slice, axis=-1)
		r = tf.math.divide_no_nan(s, x)
		r_arr.append(r)
	r_arr = tf.stack(r_arr)
	sigma = tf.reduce_mean(r_arr, axis=(0,1))
	return sigma

# def branch_param(spikes, lag=0, tau=1):
#
# 	trial_idx, seq_len, units = spikes.shape
# 	a = np.sum(spikes, axis=-1)
# 	r_arr = []
# 	for t in range(lag+1, seq_len-tau-lag-1):
# 		slice = a[:,t+lag:t+lag+tau]
# 		x = a[:, t]
# 		s = np.sum(slice, axis=-1)
# 		r = np.divide(s, x, out=np.zeros_like(s), where=x!=0)
# 		r_arr.append(r)
# 	r_arr = np.array(r_arr)
# 	sigma = np.mean(r_arr, axis=(0,1))
# 	return np.round(sigma, 3)
