import matplotlib.pyplot as plt
from matplotlib import cm
import powerlaw
import numpy as np

def fit_power(spikes):

    iters, batches, seq_len, units = spikes.shape
    map = cm.get_cmap('coolwarm')
    colors = map(np.linspace(0, 1, iters))
    #look at the last batch of the last iteration
    for iter in range(iters):
        spike_count = np.sum(spikes[iter][-1].T, axis=0)
        zero_idx = np.argwhere(spike_count == 0).flatten()
        split_spike_count = np.split(spike_count, zero_idx)
        av_sizes = np.array([np.sum(x) for x in split_spike_count])
        vals, edges = np.histogram(av_sizes, bins=10)
        plt.plot(vals, color=colors[iter])
    plt.show()

    #fit = powerlaw.Fit(av_sizes)
    #edges, prob = fit.pdf()
    #exp = fit.truncated_power_law.parameter1


def plot1(spikes, l1, l2, rates, branching, input):

    fig, ax = plt.subplots(3, 2)
    ax[0,0].imshow(spikes[-1][-1].T, cmap='gray')
    ax[0,0].set_xlabel('Time (ms)')
    ax[0,0].set_ylabel('Spikes')

    ax[1,0].plot(l1)
    ax[1,0].set_xlabel('Iteration')
    ax[1,0].set_ylabel('L1')

    ax[2,0].plot(l2)
    ax[2,0].set_xlabel('Iteration')
    ax[2,0].set_ylabel('L2')

    ax[0,1].hist(1e3*rates[-1])
    ax[0,1].set_xlabel('Firing Rate (Hz)')
    ax[0,1].set_ylabel('PDF')

    ax[1,1].plot(branching)
    ax[1,1].set_xlabel('Iteration')
    ax[1,1].set_ylabel('Branching')

    # ax[2,1].plot(av_size_hist)
    # ax[2,1].set_xscale('log')
    # ax[2,1].set_yscale('log')
    # ax[2,1].set_xlabel('Avalanche size')
    # ax[2,1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()
