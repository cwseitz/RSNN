import numpy as np
from hebb.learning_rule import *
from hebb.connectivity import *
from hebb.transfer_function import *
from hebb.network_dynamics import *
from hebb.load import load
import multiprocessing as mt
import matplotlib.pyplot as plt
import os

#fixed-point or chaotic attractors as retrival state
# TypeDynamics = 'chaos'
TypeDynamics = 'fixedpoint'
tf_params, lr_params, amp_median = load()

if  TypeDynamics =='chaos':
	amp_median *= 3
	num_patterns =int(0.56 * 250)#70-80
else:
	amp_median = amp_median
	num_patterns = 30

iters = 10
units = 500
c = 0.005
t1, t2, t3 = 2500,500,6000

#training stimulus
train_stim = np.random.normal(0.,1.,size=(num_patterns,units))

#testing stimulus
test_stim = np.random.normal(0.,1.,size=(num_patterns,units))

#init objects
tf = TransferFunction(tf_params)
lr = LearningRule(lr_params,tf)
conn = ConnectivityMatrix(lr,tf,units,c,num_patterns, random_seed=5)
mat = conn.train(train_stim)

# x = np.linspace(-10, 10, 1000)
# fig, ax = plt.subplots(1,3)
# ax[0].hist(train_stim[0])
# ax[2].hist(tf.TF(train_stim[0]))
# ax[1].plot(x, tf.TF(x))
# plt.tight_layout()
# plt.show()

for i in range(10):

	print(f'Running iteration {i}')

	#zero stimulus
	zero_stim = np.zeros((units,))

	init = np.random.normal(0,1,units)
	#background period
	rates1 = simulate(tf,mat,zero_stim,init,period=t1)
	#presentation period
	rates2 = simulate(tf,mat,train_stim[0],rates1[-1,:],period=t2)
	#delay period
	rates3 = simulate(tf,mat,zero_stim,rates2[-1,:],period=t3)

	all_rates = np.concatenate((rates1, rates2, rates3),axis=0)

	plt.plot(all_rates[:, :10])
	plt.show()

	plt.hist(all_rates[-1, :])
	plt.show()
