import numpy as np
from scipy import sparse

def simulate(tf,conn,stim,period=100,dt=0.5,tau=20):

	units = conn.shape[0]
	time_steps = int(period/dt)
	rates = np.zeros((time_steps, units))
	rates[0, :] =  np.random.normal(0,1,units)

	for step in range(time_steps):
		rec = conn.dot(rates[step-1, :])
		rates[step, :] = (dt/tau)*(-rates[step-1, :] + tf.TF(stim + rec))

	return rates
