import numpy as np
from scipy.stats import bernoulli
from scipy import sparse
import time

class ConnectivityMatrix:
	'''This class creates the connectivity matrix'''

	def __init__(self,lr,tf,units,c,num_patterns,random_seed=7):
		np.random.seed(random_seed) # fixed the seed

		#tranfer function and learning rule
		self.myTF=tf
		self.myLR=lr

		# parameters for the dynamics
		self.N=int(units)
		self.c=c
		self.p=int(num_patterns)

	def train(self, train_stim):

		self.patterns_fr = self.myTF.TF(train_stim)

		patterns_pre=self.myLR.g(self.patterns_fr)
		patterns_post=self.myLR.f(self.patterns_fr)

		rv=bernoulli(1).rvs
		indexes=sparse.find(sparse.random(self.N,self.N,density=self.c,data_rvs=rv))

		row_ind=indexes[0]
		column_ind=indexes[1]
		N2bar = len(indexes[1])
		#print 'Structural connectivity created'

		dN=1
		n=int(N2bar/dN)
		connectivity=np.array([])
		for l in range(n):
			# fast way to write down the outer product learning
			con_chunk=np.einsum('ij,ij->j',patterns_post[:,row_ind[l*dN:(l+1)*dN]],patterns_pre[:,column_ind[l*dN:(l+1)*dN]])
			connectivity=np.concatenate((connectivity,con_chunk),axis=0)
			#print 'Synaptic weights created:',100.*(l)/float(n),'%'
		con_chunk=np.einsum('ij,ij->j',patterns_post[:,row_ind[n*dN:N2bar]],patterns_pre[:,column_ind[n*dN:N2bar]])
		#print 'Synaptic weights created:',100.,'%'
		connectivity=np.concatenate((connectivity,con_chunk),axis=0)
		connectivity=(self.myLR.Amp/(self.c*self.N))*connectivity
		#print 'Synaptic weights created'

		connectivity=sparse.csr_matrix((connectivity,(row_ind,column_ind)),shape=(self.N,self.N))
		#print 'connectivity created'

		return connectivity
