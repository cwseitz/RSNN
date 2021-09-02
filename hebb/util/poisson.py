import numpy as np

def poisson_input(time, dt, units=100, rates=None):

    """

    Generate an ensemble of Poisson spike trains from a vector of firing rates

    Parameters
    ----------
    rates : ndarray
        A matrix - each value providing the firing rate for a single input unit
        If not specified, generate an ensemble of homogeneous poisson processes
        with rate parameter equal to 0.1

    Returns
    -------
    spikes : 2d ndarray
        a binary matrix containing the spiking patterns for the ensemble
        (i, j) --> (unit, time)

    """

    nsteps = int(round(time/dt))

    if rates is None:
        rates = 0.5*np.ones((units, nsteps))

    rates = rates*dt
    x = np.random.uniform(0,1,size=(units,nsteps))
    spikes = np.array(x < rates, dtype=np.int32)

    return spikes
