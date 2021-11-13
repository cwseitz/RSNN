import numpy as np
import hebb_backend

################################################################################
## Author: Clayton Seitz
## Copyright: 2021, The Hebb Project
## Email: cwseitz@uchicago.edu
################################################################################

class RNN:

    def __init__(self, N, T, dt, trials, Nrecord):

        """
        RNN base class

        Parameters
        ----------
        """

        #Basic parameters common to all RNNs
        self.N = N
        self.T = T #simulation period
        self.dt = dt #time resolution
        self.trials = trials #number of trials
        self.Nt = int(round((self.T/dt)))
        self.Nrecord = Nrecord
        self.Irecord = list(np.random.randint(0,self.Nrecord,size=(Nrecord,)))
        self.Irecord = [x.item() for x in self.Irecord] #convert to native type
        self.shape = (self.N,self.trials,self.Nt)

class ExInEIF(RNN):

    def __init__(self,N,trials,Nrecord,T,Nt,N_e,N_i,q,dt,pee0,pei0,pie0,pii0,jee,jei,
                 jie,jii,wee0,wei0,wie0,wii0,Kee,Kei,Kie,Kii,taux,
                 tausyne,tausyni,tausynx,Jee,Jei,Jie,Jii,maxns,gl,Cm,vlb,vth,
                 DeltaT,vT,vl,vre,tref, mxe, mxi, vxe, vxi):

        """

        Wrapper object for exponential integrate-and-fire (EIF) RNN in C

        Parameters
        ----------
        See comments below

        """

        super(ExInEIF, self).__init__(N, T, dt, trials, Nrecord)
        self.V = []; self.I_e = []; self.I_i = []; self.spikes = []
        self.ffwd = []

        #Excitatory-inhibitory params
        self.N_e = N_e #number of excitatory neurons
        self.N_i = N_i #number of inhibitory neurons
        self.q = q #Fraction of neurons that are excitatory
        self.maxns=maxns #maximum number of spikes in a simulation

        #Connectivity params
        self.pee0=pee0 #probability of E-E connection ~ O(1)
        self.pei0=pei0 #probability of E-I connection ~ O(1)
        self.pie0=pie0 #probability of I-E connection ~ O(1)
        self.pii0=pii0 #probability of I-I connection ~ O(1)
        self.jee=jee #psp of E-E connection ~ O(1)
        self.jei=jei #psp of E-I connection ~ O(1)
        self.jie=jie #psp of I-E connection ~ O(1)
        self.jii=jii #psp of I-I connection ~ O(1)
        self.wee0 = self.jee*self.pee0*self.q #weight of E-E connection ~ O(1)
        self.wei0 = self.jei*self.pei0*(1-self.q) #weight of E-E connection ~ O(1)
        self.wie0 = self.jie*self.pie0*self.q #weight of E-E connection ~ O(1)
        self.wii0 = self.jii*self.pii0*(1-self.q) #weight of E-E connection ~ O(1)
        self.Kee=Kee #number of postsynaptic E neurons for an E neuron ~ O(N)
        self.Kei=Kei #number of postsynaptic E neurons for an I neuron ~ O(N)
        self.Kie=Kie #number of postsynaptic I neurons for an E neuron ~ O(N)
        self.Kii=Kii #number of postsynaptic I neurons for an I neuron ~ O(N)
        self.Jee=Jee #scaled psp of E-E connection ~ O(1/sqrt(N))
        self.Jei=Jei #scaled psp of E-E connection ~ O(1/sqrt(N))
        self.Jie=Jie #scaled psp of E-E connection ~ O(1/sqrt(N))
        self.Jii=Jii #scaled psp of E-E connection ~ O(1/sqrt(N))

        #Neuron params
        self.tausyne=tausyne #time constant for epsp
        self.tausyni=tausyni #time constant for ipsp
        self.tausynx=tausynx #time constant for external psp (if input is spikes)
        self.taux=taux

        self.gl=gl #leak conductance
        self.Cm=Cm #membrane capacitance
        self.vlb=vlb #lower bound on the voltage
        self.vth=vth #threshold
        self.DeltaT=DeltaT #sharpness of AP generation
        self.vT=vT #intrinsic membrane threshold
        self.vl=vl #reversal potential for leak
        self.vre=vre #resting potential
        self.tref=tref #duration of the refractory period

        self.mxe = mxe
        self.mxi = mxi
        self.vxe = vxe
        self.vxi = vxi


    def call(self, v0):

        #Construct parameter list for call to C backend

        params = [self.N,self.Nrecord,self.T,self.Nt,self.N_e,self.N_i,self.q,
        self.dt,self.pee0,self.pei0,self.pie0,self.pii0,self.jee,self.jei,
        self.jie,self.jii,self.wee0,self.wei0,self.wie0,self.wii0,self.Kee,
        self.Kei,self.Kie,self.Kii,self.taux,self.mxe,self.mxi,self.vxe,
        self.vxi,self.tausyne,self.tausyni,self.tausynx,self.Jee,self.Jei,
        self.Jie,self.Jii,self.maxns,self.gl,self.Cm,self.vlb,self.vth,self.DeltaT,
        self.vT,self.vl,self.vre,self.tref,self.Nrecord,self.Irecord,v0]


        for i in range(self.trials):
            ctup = hebb_backend.EIF(params)
            self.add_trial(ctup)

        self.V = np.array(self.V)
        self.V = np.swapaxes(np.swapaxes(self.V,1,2),0,1)
        self.I_e = np.swapaxes(np.swapaxes(self.I_e,1,2),0,1)
        self.I_i = np.swapaxes(np.swapaxes(self.I_i,1,2),0,1)
        self.ffwd = np.swapaxes(np.swapaxes(self.ffwd,1,2),0,1)
        self.spikes = np.swapaxes(self.spikes,0,1)

    def add_trial(self, tup):

        s, v, i_e, i_i, i_x, ffwd = tup

        nspikes_record = 1000 #not recording spikes from same neurons as currents
        trial_spikes = np.zeros((nspikes_record, self.Nt), dtype=np.bool)
        for unit in range(nspikes_record):
            slice = s[s[:,1] == unit]
            spike_times = slice[:,0]
            for time in spike_times:
                trial_spikes[unit,int(round(time/self.dt))] = 1

        self.V.append(v)
        self.I_e.append(i_e)
        self.I_i.append(i_i)
        self.ffwd.append(ffwd)
        self.spikes.append(trial_spikes)

# class RNN:
#
#     def __init__(self, T, dt, tau_ref, J, trials=1, dtype=np.float32):
#
#         """
#
#         RNN base class
#
#         Parameters
#         ----------
#
#         T: float
#             Total simulation time in seconds
#         dt: float
#             Time resolution in seconds
#         tau_ref : float
#             Refractory time in seconds
#         J : 2D ndarray
#             Synaptic connectivity matrix
#         trials : int
#             Number of stimulations to run
#         dtype : numpy data type
#             Data type to use for neuron state variables
#
#         """
#
#         #Basic parameters common to all neurons
#         self.dt = dt #time resolution
#         self.T = T #simulation period
#         self.trials = trials #number of trials
#         self.tau_ref = tau_ref #refractory period
#         self.nsteps = 1 + int(round((self.T/dt))) #number of 'cuts'
#         self.ref_steps = int(self.tau_ref/self.dt) #number of steps for refractory period
#         self.J = J #synaptic connectivity
#         self.N = self.J.shape[0]
#         self.dtype = dtype #data type
#         self.shape = (self.N,self.trials,self.nsteps)

# class ClampedLIF(RNN):
#
#     def __init__(self,  T, dt, tau_ref, J, trials=1, tau=0.02, g_l=8.75, thr=20, dtype=np.float32):
#
#         super(ClampedLIF, self).__init__(T, dt, tau_ref, J=J, trials=trials, dtype=dtype)
#
#         """
#
#         Leaky Integrate & Fire (LIF) neuron model where a subset of the
#         neurons are clamped to user specified spike trains. This is useful
#         when you want an 'input population' to be part of the larger network
#
#         Parameters
#         ----------
#
#         T: float
#             Total simulation time in seconds
#         dt: float
#             Time resolution in seconds
#         tau_ref : float
#             Refractory time in seconds
#         J : 2D ndarray
#             Synaptic connectivity matrix
#         trials : int
#             Number of stimulations to run
#         tau : float
#             Membrane time constant
#         g_l : float
#             The leak conductance of the membrane
#         thr : float
#             Firing threshold in mV
#         dtype : numpy data type
#             Data type to use for neuron state variables
#
#         """
#
#         #ClampedLIF specific parameters
#         self.tau = tau
#         self.g_l = g_l
#         self.thr = thr
#
#     def spike_function(self, v):
#         z = (v >= self.thr).astype('int')
#         return z
#
#     def zero_state(self):
#
#         #Initialize state variables
#         self.I = np.zeros(shape=(self.M, self.trials, self.nsteps), dtype=self.dtype)
#         self.V = np.zeros(shape=(self.M, self.trials, self.nsteps), dtype=self.dtype)
#         self.R = np.zeros(shape=(self.M,self.trials,self.nsteps+self.ref_steps), dtype=np.bool)
#         self.Z = np.zeros(shape=(self.N, self.trials, self.nsteps), dtype=np.bool)
#
#     def call(self, spikes, clamp_idx):
#
#         """
#
#         This function will infer the indices of clamped neurons based
#         on the first axis of 'spikes'. The user is responsible for
#         determining which neurons are clamped (e.g., random, a group, etc.)
#
#         spikes : 3D ndarray
#             Used to clamp the observable state Z of specified neurons, often
#             to use a subnetwork as an 'input population'.
#         clamp_idx : 3D ndarray
#             Indices along first axis indicating which neurons are clamped
#         """
#
#         self.spikes = spikes
#         self.clamp = np.zeros((self.N,))
#         self.clamp[clamp_idx] = 1
#         self.clamp = np.mod(self.clamp + 1,2) #invert clamp (see usage below)
#         self.no_clamp_idx = np.argwhere(self.clamp > 0)
#         self.no_clamp_idx = self.no_clamp_idx.reshape((self.no_clamp_idx.shape[0],))
#         self.M = len(self.no_clamp_idx)
#         self.J = self.J[self.no_clamp_idx,:]
#         self.zero_state()
#
#         start, end = 1, self.nsteps
#
#         for i in range(start, end):
#
#             #enforce the clamp
#             self.Z[:,:,i-1] = np.einsum('ij,i -> ij', self.Z[:,:,i-1], self.clamp) + self.spikes[:,:,i-1]
#             self.I[:,:,i] =  np.matmul(self.J, self.Z[:,:,i-1])
#             #apply spike function to previous time step
#             self.Z[self.no_clamp_idx,:,i] = self.spike_function(self.V[:,:,i-1])
#             #check if the neuron spiked in the last tau_ref time steps
#             self.R[:,:,i+self.ref_steps] = np.sum(self.Z[self.no_clamp_idx,:,i-self.ref_steps:i+1], axis=-1)
#             self.V[:,:,i] = self.V[:,:,i-1] - self.dt*self.V[:,:,i-1]/self.tau +\
#                             self.I[:,:,i-1]/(self.tau*self.g_l)
#             #Enforce refractory period
#             self.V[:,:,i] = self.V[:,:,i] - self.V[:,:,i]*self.R[:,:,i+self.ref_steps]
#
# class LIF(RNN):
#
#     def __init__(self, T, dt, tau_ref, v0, J, trials, tau, thr, dtype=np.float16):
#
#         super(LIF, self).__init__(T, dt, tau_ref, J=J, trials=trials, dtype=dtype)
#
#         """
#         Basic Leaky Integrate & Fire (LIF) neuron model. For use when the
#         input currents to each neuron (from an external input pop) are known.
#         To generate currents from spikes and an input connectivity matrix,
#         see utility functions.
#
#         Parameters
#         ----------
#
#         T: float
#             Total simulation time in seconds
#         dt: float
#             Time resolution in seconds
#         tau_ref : float
#             Refractory time in seconds
#         v0 : float
#             Resting potential
#         J : 2D ndarray
#             Synaptic connectivity matrix
#         trials : int
#             Number of stimulations to run
#         tau : float
#             Membrane time constant
#         thr : float
#             Firing threshold
#         dtype : numpy data type
#             Data type to use for neuron state variables
#
#         """
#
#         #LIF specific parameters
#         self.tau = tau
#         self.thr = thr
#         self.v0 = v0
#
#     def spike_function(self, v):
#         z = (v >= self.thr).astype('int')
#         return z
#
#     def zero_state(self):
#
#         #Initialize state variables
#         self.I_r = np.zeros(shape=(self.N, self.trials, self.nsteps), dtype=self.dtype)
#         self.V = np.zeros(shape=(self.N, self.trials, self.nsteps), dtype=self.dtype)
#         self.Z = np.zeros(shape=(self.N, self.trials, self.nsteps), dtype=np.int8)
#         self.V[:,:,0] = self.v0
#
#     def check_shape(self, x):
#         if x is None:
#             raise ValueError('Input object was not set')
#         else:
#             if x.shape != self.shape:
#                 raise ValueError('Check input object shape')
#         return True
#
#     def call(self, ffwd):
#
#         self.ffwd = ffwd
#         self.check_shape(self.ffwd)
#         self.zero_state()
#
#         for i in range(1,self.nsteps):
#
#             if i % 100 == 0:
#                 print(f'Time step {i}')
#
#             i_in = self.ffwd[:,:,i-1]
#             self.I_r[:,:,i] = np.matmul(self.J, self.Z[:,:,i-1])
#
#             # #update neuron voltages
#             # self.V[:,:,i] = self.V[:,:,i-1] - self.dt*self.V[:,:,i-1]/self.tau +\
#             #                 self.ffwd[:,:,i-1] + self.I_r[:,:,i-1]
#
#             self.V[:,:,i] = self.V[:,:,i-1] + (self.ffwd[:,:,i] + self.I_r[:,:,i])*(self.dt/self.tau)
#
#             #find the neurons which spiked in the last ref_steps time steps
#             if i > self.ref_steps:
#                 ref = np.sum(self.Z[:,:,i-self.ref_steps:i],axis=-1)
#                 #set voltages to zero if refractory
#                 self.V[:,:,i] = self.V[:,:,i]*(1-ref)
#             else:
#                 ref = np.sum(self.Z[:,:,:self.ref_steps],axis=-1)
#                 #set voltages to zero if refractory
#                 self.V[:,:,i] = self.V[:,:,i]*(1-ref)
#
#             #determine which neurons cross threshold in this time step
#             self.Z[:,:,i] = self.spike_function(self.V[:,:,i])
#             #reset the voltage if it spiked
#             self.V[:,:,i] = self.V[:,:,i]*(1-self.Z[:,:,i])
