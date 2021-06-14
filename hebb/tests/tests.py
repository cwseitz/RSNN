from ..models import *
from ..util import *

def tf_gpu_test():
    pass

def check_ex_in_conn_mat():

    n_excite = 80; n_inhib = 20; n_input = 10
    p_ee, p_ei, p_ie, p_ii = 0.2, 0.3, 0.35, 0.25
    mu = -0.64; sigma = 0.51

    rec_cmg = ExInConnectivityMatrixGenerator(n_excite, n_inhib, p_ee, p_ei, p_ie, p_ii, mu, sigma)
    rec_cmg.run_generator()

    in_cmg = InputConnectivityGenerator(n_input, n_excite+n_inhib)
    in_cmg.run_generator()

    weight_plot(in_cmg, rec_cmg)
    plt.show()
