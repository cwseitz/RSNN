


def unit_v_stats(cell, dv=0.01):

    """
    Compute the histogram of voltage values for a single neuron over
    trials, as a function of time i.e. P(V,t)
    The vector over which P is calculated has shape (1, trials, 1)
    """

    bins = np.arange(0, cell.thr, dv)
    temp = np.zeros((cell.nsteps,480,640,3))
    imsave('data/temp.tif', temp)
    im = pims.open('data/temp.tif')

    h = np.apply_along_axis(lambda a: np.histogram(a, bins=bins, density=True)[0], 1, cell.V)
    for t in range(cell.nsteps):
        fig, ax = plt.subplots()
        ax.imshow(h[:,:,t], cmap='coolwarm')
        rgb_array_3d = plt2array(fig)
        im[t] = rgb_array_3d

def add_rate_hist(ax, cell, bins=20):

    rates = np.mean(cell.Z,axis=1)
    fig, ax = plt.subplots()
    bins = np.linspace(rates.min(), rates.max(), bins)
    colors = cm.coolwarm(np.linspace(0,1,cell.nsteps))
    for t in range(cell.nsteps):
        #idx = np.nonzero(clamp[:,0,t])
        vals, bins = np.histogram(rates[:,t], bins=bins)
        ax.plot(bins[:-1], vals, color=colors[t])

def add_v_stats(ax, cell, dv=0.05):

    """
    Compute the histogram of voltage values over a population
    as a function of time i.e. P(V,t)
    """

    bins = np.arange(0, cell.thr, dv)
    fig, ax = plt.subplots()
    colors = cm.coolwarm(np.linspace(0,1,cell.nsteps))
    for t in range(cell.nsteps):
        #idx = np.nonzero(cell.clamp[:,0,t])
        vals, bins = np.histogram(cell.V[:,:,t], bins=bins)
        vals = vals/(np.sum(vals)*dv)
        ax.plot(bins[:-1], vals, color=colors[t])
