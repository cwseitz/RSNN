import numpy as np
import matplotlib.pyplot as plt

mu = 0
sigma = 1

for i in range(100):
    mu -= 0.1
    #sigma += 0.1
    sample = np.random.lognormal(mu, sigma, size=100)
    plt.hist(sample)
    plt.xlim([0,2])
    plt.title(f'{mu}')
    plt.show()
