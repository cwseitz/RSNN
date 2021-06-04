import numpy as np
import pickle

def load():

    f = open('../hebb/parametersFit.p','rb')
    paramfit = pickle.load(f, encoding='latin1')

    # using the median parameters of the fits
    rmax_median = np.median(paramfit[0][0])
    beta_median = np.median(paramfit[0][1])
    h0_median = np.median(paramfit[0][2])
    tf_params = ['sig',rmax_median,beta_median,h0_median] # param TF

    amp_median = np.median(paramfit[1][0])
    qf = np.median(paramfit[1][1])#0.65
    bf_median = np.median(paramfit[1][2])
    xf = np.median(paramfit[1][3])#22.

    lr_params = [xf,xf,bf_median,bf_median,qf,amp_median]  #learning rule

    return tf_params, lr_params, amp_median
