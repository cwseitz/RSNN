import numpy as np

################################################################################
## Author: Clayton Seitz
## Copyright: 2021, The Hebb Project
## Email: cwseitz@uchicago.edu
################################################################################

class FFWD_EIF:

    def __init__(self, N, Nt, mxe, mxi, vxe, vxi, taux, rxe, rxi, jeX, jiX):

        """
        Feedforward input for EIF simulation

        Parameters
        ----------
        """

        #Stimulus params
        self.mxe=mxe #Mean value of the excitatory stimulus
        self.mxi=mxi #Mean value of the inhibitory stimulus
        self.rxe = rxe
        self.rxi = rxi
        self.jeX = jeX
        self.jiX = jiX
        self.mxe0=(self.mxe/np.sqrt(N))+self.rxe*self.jeX/N
        self.mxi0=(self.mxi/np.sqrt(N))+self.rxi*self.jiX/N

        self.taux=taux #timescale for autocorrelation of external stimulus
        self.vxe=vxe #Variance of the excitatory stimulus
        self.vxi=vxi #Variance of the inhibitory stimulus
        self.s_e = np.random.normal(0,np.sqrt(vxe),size=(Nt,))
        self.s_i = np.random.normal(0,np.sqrt(vxi),size=(Nt,))

        self.Ix1e = list(self.mxe+self.s_e) #Stimulus for first fraction of E neurons
        self.Ix1i = list(self.mxi+self.s_i) #Stimulus for second fraction of E neurons
        self.Ix2e = list(self.mxe+self.s_e) #Stimulus for first fraction of I neurons
        self.Ix2i = list(self.mxi+self.s_i) #Stimulus for second fraction of I neurons
