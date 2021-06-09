import matplotlib.pyplot as plt
from matplotlib import cm
import powerlaw
import numpy as np

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

def avalanche_per_iter(spikes):

    iters, batches, seq_len, units = spikes.shape
    map = cm.get_cmap('coolwarm')
    colors = map(np.linspace(0, 1, iters))

    for iter in range(iters):
        avalanche_plot(spikes, iter=iter, color=colors[iter])

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

def weight_plot(in_weights, rec_weights):

    fig, ax = plt.subplots(2, 3)
    ax[0,0].imshow(in_weights.conn, cmap='gray')
    ax[0,0].set_title('Input connectivity', size=8)
    ax[0,0].set_xlabel('Input')
    ax[0,0].set_ylabel('Neuron')

    ax[0,1].imshow(in_weights.weights, cmap='gray')
    ax[0,1].set_title('Input weights', size=8)
    ax[0,1].set_xlabel('Input')
    ax[0,1].set_ylabel('Neuron')

    ax[1,0].imshow(rec_weights.conn, cmap='gray')
    ax[1,0].set_title('Recurrent connectivity', size=8)
    ax[1,0].set_xlabel('Neuron')
    ax[1,0].set_ylabel('Neuron')

    ax[1,1].imshow(rec_weights.weights, cmap='gray')
    ax[1,1].set_title('Recurrent weights', size=8)
    ax[1,1].set_xlabel('Neuron')
    ax[1,1].set_ylabel('Neuron')

    in_partition_ind = [(0, in_weights.conn.shape[0]),(0, in_weights.conn.shape[1])]
    flat_in = vectorize(in_weights.conn, in_weights.weights, in_partition_ind)
    in_vals, in_bins = np.histogram(flat_in, density=True)

    ax[0,2].plot(in_bins[:-1], in_vals, color='red', label='Input')
    ax[0,2].set_xlabel('Synaptic weight (mV)')
    ax[0,2].set_ylabel('Density')
    ax[0,2].legend()

    ex_part_ind = [(0, rec_weights.n_excite), (0, rec_weights.conn.shape[1])]
    flat_ex = vectorize(rec_weights.conn, rec_weights.weights, ex_part_ind)

    inh_part_ind = [(rec_weights.n_excite, rec_weights.conn.shape[0]),
                   (0, rec_weights.conn.shape[1])]

    flat_inh = vectorize(rec_weights.conn, rec_weights.weights, inh_part_ind)

    ex_vals, ex_bins = np.histogram(flat_ex, bins=30, density=True)
    inh_vals, inh_bins = np.histogram(flat_inh, bins=30, density=True)

    #ax[1,2].plot(inh_bins[:-1], inh_vals, color='red', label='Inhibitory')
    ax[1,2].plot(ex_bins[:-1], ex_vals, color='blue', label='Excitatory')
    ax[1,2].set_xlabel('Synaptic weight (mV)')
    ax[1,2].set_ylabel('Density')
    ax[1,2].legend()

    plt.tight_layout()


def spike_plot(input, spikes):

    input_activity = get_counts_per_frame(input[-1])
    res_activity = get_counts_per_frame(spikes[-1])

    fig, ax = plt.subplots(4,1, sharex=True)
    ax[0].imshow(input[-1].T, cmap='gray')
    ax[0].set_ylabel('Spikes')

    ax[1].plot(input_activity, color='red')
    ax[1].set_ylabel('Input activity')

    ax[2].imshow(spikes[-1].T, cmap='gray')
    ax[2].set_ylabel('Reservoir spikes')

    ax[3].plot(res_activity, color='blue')
    ax[3].set_ylabel('Reservoir activity')
    ax[3].set_xlabel('Time (ms)')

    plt.tight_layout()

def train_plot(l1, l2, rates, branching):

    fig, ax = plt.subplots(2, 2)

    ax[1,0].plot(l1, color='red', label='L1')
    ax[1,0].set_xlabel('Iteration')
    ax[1,0].set_ylabel('Loss')

    ax[1,0].plot(l2, color='blue', label='L2')
    ax[1,0].set_xlabel('Iteration')
    ax[1,0].legend()

    ax[0,1].hist(1e3*rates, color='red')
    ax[0,1].set_xlabel('Firing Rate (Hz)')
    ax[0,1].set_ylabel('PDF')

    ax[1,1].plot(branching, color='black')
    ax[1,1].set_xlabel('Iteration')
    ax[1,1].set_ylabel('Branching')



    plt.tight_layout()
    plt.show()
