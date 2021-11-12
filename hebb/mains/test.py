import hebb_backend
import matplotlib.pyplot as plt

x = hebb_backend.multi_gauss_ind(5000,5.0,2.0)
plt.hist(x)
plt.show()
