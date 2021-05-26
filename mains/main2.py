import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy import integrate
from scipy.optimize import brentq

def sigmoid(x, a, b, x0):
	return a/(1+np.exp(-b*(x-x0)))

def f(x, qf, betaf, xf):
    return 0.5*(2*qf-1.+np.tanh(betaf*(x-xf)))

def g(x, qg, betag, xg):
    return 0.5*(2*qg-1.+np.tanh(betag*(x-xg)))

def get_qg():
    return brentq(eg,0.,1.)

def eg(q):
    qg=q
    fun=lambda x:std_normal(x)*g(sigmoid(x))
    var,err=integrate.quad(fun,-10.,10.)
    return var

def std_normal(x):
    sigma=1.; mu=0
    pdf=(1./np.sqrt(2 * np.pi * sigma**2))*np.exp(-(1./2.)*((x-mu)/sigma)**2)
    return pdf

file = open('../hebb/parametersFit.p','rb')
paramfit = pickle.load(file, encoding='latin1')

#using the median parameters of the fits
rmax = np.median(paramfit[0][0])
betaf = np.median(paramfit[0][1])
h0 = np.median(paramfit[0][2])

amp = np.median(paramfit[1][0])
qf = np.median(paramfit[1][1])
bf = np.median(paramfit[1][2])
xf = np.median(paramfit[1][3])

xg = xf 
betag = betaf
qg = get_qg()


n = 1000
mean = np.zeros((n,))
cov = np.eye(n)
x = np.random.multivariate_normal(mean, cov)
y = sigmoid(x, rmax, bf, h0)


rate_pre = f(y, qf, betaf, xf)
rate_post = g(y, qg, betag, xg)

