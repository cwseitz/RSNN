import numpy as np

def multi_gauss(r, mu, sigma):

    """
    A multivariate gaussian defined over a discrete space r
    This is NOT a probability distribution and therefore is not normalized

    To make it a pdf the normalization constant is 1/np.sqrt((2*np.pi)**n * det)

    """

    n = mu.shape[0]
    det = np.linalg.det(sigma)
    inv = np.linalg.inv(sigma)
    fac = np.einsum('...k,kl,...l->...', r-mu, inv, r-mu)
    return np.exp(-fac / 2)

def prod_gauss(mu_1, sigma_1, mu_2, sigma_2):

    """
    Return the gaussian parameters of a product of multivariate gaussians
    """

    mu_3 = sigma_2 @ np.linalg.inv(sigma_1+sigma_2) @ mu_1 +\
           sigma_1 @ np.linalg.inv(sigma_1+sigma_2) @ mu_2

    sigma_3 = sigma_1 @ np.linalg.inv(sigma_1+sigma_2) @ sigma_2

    return mu_3, sigma_3

def delta_gauss(dx, sigma, delta):

    """
    Return the value of a gaussian a distance dx from the mean
    """

    # a = (1/(sigma*np.sqrt(2*np.pi)))
    dx *= delta
    return np.exp((-0.5*dx**2)/sigma**2)

def trinomial(k_ij, k_ji, q):

    """
    Draw a sample from a trinomial distribution to generate synapses.
    Inputs k_ij, k_ji, q should be binomial probabilities in [0,1]
    """
    p_ij, p_ji, q = _trinomial(k_ij, k_ji, q)
    x = np.random.uniform(0,1)
    if x <= p_ij:
        out = -1
    elif p_ij < x <= p_ij+p_ji:
        out = 1
    elif x > p_ij+p_ji:
        out = 0
    return out

def _trinomial(k_ij, k_ji, q):

    p_ij = k_ij*(1-k_ji)*(1-q)
    p_ji = k_ji*(1-k_ij)*(1-q)
    p_x = q*(1-k_ij)*(1-k_ji)
    z_ij = p_ij + p_ji + p_x
    p_ij /= z_ij; p_ji /= z_ij; p_x /= z_ij

    return p_ij, p_ji, q


def torus_dist(r_i, r_j, M, delta):

    """
    Evaluate the Euclidean distance between two points on a discrete
    torus (a 2D lattice with periodic boundary conditions)
    """

    x1, y1 = r_i; x2, y2 = r_j
    dx = np.minimum(np.abs(x1-x2),M*delta-np.abs(x1-x2))
    dy = np.minimum(np.abs(y1-y2),M*delta-np.abs(y1-y2))
    dr = np.sqrt(dx**2 + dy**2)
    return dr

def entropy(p):

    """
    Compute the entropy of a probability distribution p
    """

    return np.sum(p*np.log2(1/p))

def hogn_avg_out_deg(N, sigma, boost, q, delta, x0=0, y0=0):

    """
    Average out degree of a homogeneous gaussian network for a given
    value of sigma, bias, and q params
    """

    M = int(round(np.sqrt(N)))
    xv, yv = np.meshgrid(np.arange(M),np.arange(M))
    X, Y = xv.ravel(), yv.ravel()
    dr_ij_vec = np.array([torus_dist((0,0),(X[i],Y[i]),M,delta) for i in range(N)])[1:]
    p_vec = np.zeros_like(dr_ij_vec)
    for i, dr_ij in enumerate(dr_ij_vec):
        k_ij = boost*delta_gauss(dr_ij, sigma, delta)
        p_ij, p_ji, q  = _trinomial(k_ij, k_ij, q)
        p_vec[i] = p_ij
    return np.sum(p_vec)

def hogn_var_out_deg(N, sigma, boost, q, delta, x0=0, y0=0):

    """
    Average out degree of a homogeneous gaussian network for a given
    value of sigma and boost params
    """

    M = int(round(np.sqrt(N)))
    xv, yv = np.meshgrid(np.arange(M),np.arange(M))
    X, Y = xv.ravel(), yv.ravel()
    dr_ij_vec = np.array([torus_dist((0,0),(X[i],Y[i]),M,delta) for i in range(N)])[1:]
    p_vec = np.zeros_like(dr_ij_vec)
    for i, dr_ij in enumerate(dr_ij_vec):
        k_ij = boost*delta_gauss(dr_ij, sigma, delta)
        p_ij, p_ji, q  = _trinomial(k_ij, k_ij, q)
        p_vec[i] = p_ij*(1-p_ij)

    return np.sum(p_vec)

def hogn_out_deg_fixsig(N, sigma, qs, bias=1, delta=1):

    """
    First two moments of the out the degree distribution for fixed
    sigma, varying the bias and sparsity parameters

    Will compute suitable values for the bias parameter based on sigma.
    """

    avg_arr = np.zeros((len(qs),))
    var_arr = np.zeros((len(qs),))
    for i,q in enumerate(qs):
        avg_arr[i] = hogn_avg_out_deg(N, sigma, bias, q, delta)
        var_arr[i] = hogn_var_out_deg(N, sigma, bias, q, delta)
    return avg_arr, var_arr

def hogn_out_deg_fixq(N, sigmas, q, bias=1, delta=1):

    """
    First two moments of the out degree distribution for fixed
    sparsity parameter, varying the reach (sigma) parameter
    """

    avg_arr = np.zeros((len(sigmas),))
    var_arr = np.zeros((len(sigmas),))
    for i,sigma in enumerate(sigmas):
        avg_arr[i] = hogn_avg_out_deg(N, sigma, bias, q, delta)
        var_arr[i] = hogn_var_out_deg(N, sigma, bias, q, delta)
    return avg_arr, var_arr


def hogn_out_deg_full(N, sigmas, qs, bias=1, delta=1):

    """
    Mean of the out the degree distribution over the entire parameter space
    for the homogeneous gaussian network (sigma, q)
    """

    avg_arr = np.zeros((len(qs),len(sigmas)))
    var_arr = np.zeros((len(qs),len(sigmas)))
    for i,q in enumerate(qs):
        for j, sigma in enumerate(sigmas):
            avg_arr[i,j] = hogn_avg_out_deg(N, sigma, bias, q, delta)
            var_arr[i,j] = hogn_var_out_deg(N, sigma, bias, q, delta)
    return avg_arr, var_arr

def exin_avg_e_in_deg(N, sigma_e, sigma_i, bias_e, bias_i, q, p_e, delta=1):

    """
    Average number of excitatory connections and inhibitory connections coming
    into an excitatory neuron
    """

    M = int(round(np.sqrt(N)))
    xv, yv = np.meshgrid(np.arange(M),np.arange(M))
    X, Y = xv.ravel(), yv.ravel()
    dr_ij_vec = np.array([torus_dist((0,0),(X[i],Y[i]),M,delta) for i in range(N)])[1:]
    p_vec_ex = np.zeros_like(dr_ij_vec)
    p_vec_inh = np.zeros_like(dr_ij_vec)

    for i, dr_ij in enumerate(dr_ij_vec):
        k_ij = bias_e*delta_gauss(dr_ij, sigma_e, delta) #doesn't depend on postsynaptic cell-type
        k_ji_ex = p_e*bias_e*delta_gauss(dr_ij, sigma_e, delta) #incoming connection to an excitatory neuron
        k_ji_inh = (1-p_e)*bias_i*delta_gauss(dr_ij, sigma_i, delta) #incoming connection to an inhibitory neuron
        p_ij_ex, p_ji_ex, q  = _trinomial(k_ij, k_ji_ex, q)
        p_ij_inh, p_ji_inh, q  = _trinomial(k_ij, k_ji_inh, q)
        p_vec_ex[i] = p_ji_ex
        p_vec_inh[i] = p_ji_inh

    avg_in_ex = np.sum(p_vec_ex)
    avg_in_inh = np.sum(p_vec_inh)

    return avg_in_ex, avg_in_inh

def exin_avg_i_in_deg(N, sigma_e, sigma_i, bias_e, bias_i, q, p_e, delta=1):

    """
    Average number of excitatory connections and inhibitory connections coming
    into an inhibitory neuron
    """

    M = int(round(np.sqrt(N)))
    xv, yv = np.meshgrid(np.arange(M),np.arange(M))
    X, Y = xv.ravel(), yv.ravel()
    dr_ij_vec = np.array([torus_dist((0,0),(X[i],Y[i]),M,delta) for i in range(N)])[1:]
    p_vec_ex = np.zeros_like(dr_ij_vec)
    p_vec_inh = np.zeros_like(dr_ij_vec)

    for i, dr_ij in enumerate(dr_ij_vec):
        k_ij = bias_i*delta_gauss(dr_ij, sigma_i, delta) #doesn't depend on postsynaptic cell-type
        k_ji_ex = p_e*bias_e*delta_gauss(dr_ij, sigma_e, delta) #incoming connection to an excitatory neuron
        k_ji_inh = (1-p_e)*bias_i*delta_gauss(dr_ij, sigma_i, delta) #incoming connection to an inhibitory neuron
        p_ij_ex, p_ji_ex, q  = _trinomial(k_ij, k_ji_ex, q)
        p_ij_inh, p_ji_inh, q  = _trinomial(k_ij, k_ji_inh, q)
        p_vec_ex[i] = p_ji_ex
        p_vec_inh[i] = p_ji_inh

    avg_in_ex = np.sum(p_vec_ex)
    avg_in_inh = np.sum(p_vec_inh)

    return avg_in_ex, avg_in_inh
