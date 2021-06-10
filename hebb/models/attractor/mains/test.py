from hebb.network_dynamics import *
from hebb.load import *
from hebb.transfer_function import *
import numpy as np
import matplotlib.pyplot as plt

def loss_func(rates, target_rate=20):

    """
    Loss is defined as the variance of the rate over the population
    """

    rates = (rates-target_rate)**2
    loss = np.mean(rates, axis=1)
    return loss


def run():

    tf_params, lr_params, amp_median = load()
    tf = TransferFunction(tf_params)

    units = 100
    stim = np.random.normal(0.,1.,size=(units,))
    init_rates = np.random.normal(0.,1.,size=(units,))
    conn = np.random.normal(0.,1.,size=(units,units))

    rates = simulate(tf,conn,stim,init_rates, dt=0.5)
    loss = loss_func(rates)

    return conn, rates, loss

fig, ax = plt.subplots(1,2)

for i in range(1000):

    conn, rates, loss = run()
    plt.plot(loss)

plt.show()
