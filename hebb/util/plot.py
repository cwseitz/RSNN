import matplotlib.pyplot as plt
import powerlaw
import numpy as np
from matplotlib import cm
from .util import *


def activity_plot(input, spikes):

    fig, ax = plt.subplots(4,1, sharex=False)
    input_activity = get_counts_per_frame(input)
    spike_activity = get_counts_per_frame(spikes)
    input_rates = np.mean(input.T, axis=1)
    spike_rates = np.mean(spikes.T, axis=1)

    ax[0].imshow(input.T, cmap='gray')
    ax[1].imshow(spikes.T, cmap='gray')
    ax[2].plot(input_activity, color='red')
    ax[2].plot(spike_activity, color='blue')

    ax[3].hist(input_rates, color='red')
    ax[3].hist(spike_rates, color='blue')

    plt.show()


def avalanche_plot(spikes, iter=-1, batch=-1, color='blue'):

    iters, batches, seq_len, units = spikes.shape
    map = cm.get_cmap('coolwarm')
    colors = map(np.linspace(0, 1, iters))

    spikes = spikes[iter][batch]
    counts_per_frame = get_counts_per_frame(spikes)
    av_sizes = get_avalanche_sizes(spikes)

    fit = powerlaw.Fit(av_sizes)

    powerlaw.plot_pdf(av_sizes, color=color)

def weight_plot(w_ee, w_ei, w_ie, w_ii, w_e_in):

    fig, ax = plt.subplots(2, 3)

    ax[0,0].imshow(w_ee, cmap='gray')
    ax[0,0].set_title('W_EE', size=8)
    ax[0,0].set_xlabel('Neuron')
    ax[0,0].set_ylabel('Neuron')

    ax[0,1].imshow(w_ei, cmap='gray')
    ax[0,1].set_title('W_EI', size=8)
    ax[0,1].set_xlabel('Neuron')
    ax[0,1].set_ylabel('Neuron')

    ax[1,0].imshow(w_ie, cmap='gray')
    ax[1,0].set_title('W_IE', size=8)
    ax[1,0].set_xlabel('Neuron')
    ax[1,0].set_ylabel('Neuron')

    ax[1,1].imshow(w_ii, cmap='gray')
    ax[1,1].set_title('W_II', size=8)
    ax[1,1].set_xlabel('Neuron')
    ax[1,1].set_ylabel('Neuron')

    ax[1,2].imshow(w_e_in, cmap='gray')
    ax[1,2].set_title('W_IN', size=8)
    ax[1,2].set_xlabel('Input')
    ax[1,2].set_ylabel('Neuron')

    w_ee = w_ee.flatten()
    w_ei = w_ee.flatten()
    w_ie = w_ee.flatten()
    w_ii = w_ee.flatten()

    w_ee_vals, w_ee_bins = np.histogram(w_ee, bins=10, density=True)
    w_ei_vals, w_ei_bins = np.histogram(w_ei, bins=10, density=True)
    w_ie_vals, w_ie_bins = np.histogram(w_ie, bins=10, density=True)
    w_ii_vals, w_ii_bins = np.histogram(w_ii, bins=10, density=True)

    ax[0,2].plot(w_ee_bins[:-1], w_ee_vals, color='red', label='W_EE')
    ax[0,2].plot(w_ei_bins[:-1], w_ei_vals, color='blue', label='W_EI')
    ax[0,2].plot(w_ie_bins[:-1], w_ie_vals, color='black', label='W_IE')
    ax[0,2].plot(w_ii_bins[:-1], w_ii_vals, color='cyan', label='W_II')

    ax[0,2].set_xlabel('Synaptic weight (mV)')
    ax[0,2].set_ylabel('Density')
    ax[0,2].legend()

    plt.tight_layout()

# def weight_plot(in_weights, rec_weights, n_excite=80):
#
#     flat_in = in_weights.flatten()
#     in_vals, in_bins = np.histogram(flat_in, density=True)
#
#     flat_ex = rec_weights[:n_excite, :].flatten()
#     flat_inh = rec_weights[n_excite:, :].flatten()
#     ex_vals, ex_bins = np.histogram(flat_ex, bins=10, density=True)
#     inh_vals, inh_bins = np.histogram(flat_inh, bins=10, density=True)
#
#     fig, ax = plt.subplots(1, 3)
#
#     ax[0].imshow(in_weights, cmap='gray')
#     ax[0].set_title('Input weights', size=8)
#     ax[0].set_xlabel('Neuron')
#     ax[0].set_ylabel('Input')
#
#     ax[1].imshow(rec_weights, cmap='gray')
#     ax[1].set_title('Recurrent weights', size=8)
#     ax[1].set_xlabel('Input')
#     ax[1].set_ylabel('Neuron')
#
#     ax[2].plot(in_bins[:-1], in_vals, color='black', label='Input')
#     # ax[2].plot(inh_bins[:-1], inh_vals, color='blue', label='Inhibitory')
#     # ax[2].plot(ex_bins[:-1], ex_vals, color='red', label='Excitatory')
#     ax[2].set_xlim([-5, 3])
#     ax[2].set_xlabel('Synaptic weight (mV)')
#     ax[2].set_ylabel('Density')
#     ax[2].legend()
#
#     plt.tight_layout()


# def weight_plot(in_weights, rec_weights):
#
#     fig, ax = plt.subplots(2, 3)
#     ax[0,0].imshow(in_weights.conn, cmap='gray')
#     ax[0,0].set_title('Input connectivity', size=8)
#     ax[0,0].set_xlabel('Input')
#     ax[0,0].set_ylabel('Neuron')
#
#     ax[0,1].imshow(in_weights.weights, cmap='gray')
#     ax[0,1].set_title('Input weights', size=8)
#     ax[0,1].set_xlabel('Input')
#     ax[0,1].set_ylabel('Neuron')
#
#     ax[1,0].imshow(rec_weights.conn, cmap='gray')
#     ax[1,0].set_title('Recurrent connectivity', size=8)
#     ax[1,0].set_xlabel('Neuron')
#     ax[1,0].set_ylabel('Neuron')
#
#     ax[1,1].imshow(rec_weights.weights, cmap='gray')
#     ax[1,1].set_title('Recurrent weights', size=8)
#     ax[1,1].set_xlabel('Neuron')
#     ax[1,1].set_ylabel('Neuron')
#
#     in_partition_ind = [(0, in_weights.conn.shape[0]),(0, in_weights.conn.shape[1])]
#     flat_in = vectorize(in_weights.conn, in_weights.weights, in_partition_ind)
#     in_vals, in_bins = np.histogram(flat_in, density=True)
#
#     ax[0,2].plot(in_bins[:-1], in_vals, color='red', label='Input')
#     ax[0,2].set_xlabel('Synaptic weight (mV)')
#     ax[0,2].set_ylabel('Density')
#     ax[0,2].legend()
#
#     ex_part_ind = [(0, rec_weights.n_excite), (0, rec_weights.conn.shape[1])]
#     flat_ex = vectorize(rec_weights.conn, rec_weights.weights, ex_part_ind)
#
#     inh_part_ind = [(rec_weights.n_excite, rec_weights.conn.shape[0]),
#                    (0, rec_weights.conn.shape[1])]
#
#     flat_inh = vectorize(rec_weights.conn, rec_weights.weights, inh_part_ind)
#
#     ex_vals, ex_bins = np.histogram(flat_ex, bins=30, density=True)
#     inh_vals, inh_bins = np.histogram(flat_inh, bins=30, density=True)
#
#     ax[1,2].plot(inh_bins[:-1], inh_vals, color='red', label='Inhibitory')
#     ax[1,2].plot(ex_bins[:-1], ex_vals, color='blue', label='Excitatory')
#     ax[1,2].set_xlabel('Synaptic weight (mV)')
#     ax[1,2].set_ylabel('Density')
#     ax[1,2].legend()
#
#     plt.tight_layout()

def pattern_plot(spikes):

    prob = pattern_prob(spikes)
    fig, ax = plt.subplots()
    ax.plot(prob, color='red')
    ax.set_yscale('log')
    ax.set_ylabel('log P')
    ax.set_xlabel('Pattern index')
    plt.show()

def spike_plot(input, spikes, voltage):

    fig, ax = plt.subplots(3,1, sharex=True)

    ax[0].imshow(input.T, cmap='gray')
    ax[0].set_ylabel('Spikes')

    ax[1].imshow(spikes.T, cmap='gray')
    ax[1].set_ylabel('Reservoir spikes')

    ax[2].imshow(voltage.T, cmap='gray')
    ax[2].set_ylabel('Reservoir voltage')

    plt.tight_layout()

def train_plot(spike_reg_1, spike_reg_2, reg):

    fig, ax = plt.subplots()

    ax.plot(spike_reg_1, color='red', label='L1')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')

    ax.plot(spike_reg_2, color='blue', label='L2')
    ax.plot(reg, label='L3')
    ax.legend()
    ax.plot()

    # ax[0,1].hist(1e3*rates, color='red')
    # ax[0,1].set_xlabel('Firing Rate (Hz)')
    # ax[0,1].set_ylabel('PDF')
    #
    # ax[1,1].plot(branching, color='black')
    # ax[1,1].set_xlabel('Iteration')
    # ax[1,1].set_ylabel('Branching')

    plt.tight_layout()
    plt.show()
