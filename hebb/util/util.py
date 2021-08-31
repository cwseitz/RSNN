import numpy as np
# import powerlaw
import tensorflow as tf
import os
from glob import glob


def stack_tensors(save_dir='data/'):

    files = glob(save_dir + '*.npz')
    files = sorted(glob(save_dir + '*.npz'))
    files = sorted([int(os.path.basename(file).split('.')[0]) for file in files])
    files = [save_dir+str(file)+'.npz' for file in files]

    #get the names of saved arrays
    keys = load_tensors(files[0]).keys()
    tensor_dict = {k: [] for k in keys}

    for file in files:
        dict = load_tensors(file)
        for key in dict.keys():
            tensor_dict[key].append(dict[key])
    for key in keys:
        tensor_dict[key] = np.stack(tensor_dict[key])

    return tensor_dict

def save_tensors(tensor_dict, tag, save_dir='data/', name=''):
    np.savez_compressed(save_dir + name + tag, **tensor_dict)

def load_tensors(path):
    data = np.load(path)
    tensor_dict = dict(data)
    data.close()
    return tensor_dict

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

def vectorize(conn, weights, partition_ind):

    """
    Vectorize a partition of a matrix that has nonzero connectivity
    """

    row_low, row_high = partition_ind[0]
    col_low, col_high = partition_ind[1]
    part_weights = weights[row_low:row_high, col_low:col_high]
    conn_nonzero = np.nonzero(conn[row_low:row_high, col_low:col_high])
    flat_weights = part_weights[conn_nonzero].flatten()

    return flat_weights

def pattern_prob(spikes):

    new_shape = (spikes.shape[0]*spikes.shape[1], spikes.shape[-1])
    flat_spikes = np.reshape(spikes, new_shape)
    unique, counts = np.unique(flat_spikes, return_counts=True, axis=0)
    prob = np.sort(counts/np.sum(counts))[::-1]

    return prob

def get_counts_per_frame(spikes):
    return np.sum(spikes.T, axis=0)

def get_avalanche_sizes(spikes):

    av_sizes = []
    av_size = 0

    spike_count = get_counts_per_frame(spikes)
    for i in range(len(spike_count)):
        if spike_count[i] == 0 and av_size != 0:
            av_sizes.append(av_size)
            av_size = 0
        else:
            av_size += spike_count[i]

    #append if last value wasn't zero
    if av_size != 0:
        av_sizes.append(av_size)

    return av_sizes
