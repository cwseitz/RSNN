import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
from scipy import sparse
import time

class ConnectivityMatrix():

	def __init__(self,lr,tf,units,c,p,seed=7):

		np.random.seed(seed)
		self.tf=tf
		self.lr=lr

		self.units=units
		self.c=c
		self.p=int(p)
		self.value=np.array([])
		self.mask()

	def mask(self):

		rv=bernoulli(1).rvs
		self.mask = sparse.random(self.units,self.units,density=self.c,data_rvs=rv)
		self.mask_ind = sparse.find(self.mask)

	def train(self, train_stim):

		train_fr = self.tf.TF(train_stim)
		patterns_pre=self.lr.g(train_fr)
		patterns_post=self.lr.f(train_fr)

		row_ind=self.mask_ind[0]
		column_ind=self.mask_ind[1]
		N2bar = len(self.mask_ind[1]) #number of nonzero entries

		dN=1
		n=int(N2bar/dN)

		for l in range(n):
			con_chunk=np.einsum('ij,ij->j',patterns_post[:,row_ind[l*dN:(l+1)*dN]],patterns_pre[:,column_ind[l*dN:(l+1)*dN]])
			self.value=np.concatenate((self.value,con_chunk),axis=0)
			print( 'Synaptic weights created:',100.*(l)/float(n),'%')

		con_chunk=np.einsum('ij,ij->j',patterns_post[:,row_ind[n*dN:N2bar]],patterns_pre[:,column_ind[n*dN:N2bar]])
		print( 'Synaptic weights created:',100.,'%')
		self.value=np.concatenate((self.value,con_chunk),axis=0)
		self.value=(self.lr.Amp/(self.c*self.units))*self.value
		print( 'Synaptic weights created')

		self.value=sparse.csr_matrix((self.value,(row_ind,column_ind)),shape=(self.units,self.units))
		print( 'connectivity created')
