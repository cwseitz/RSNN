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

def hogn_avg_out_deg(net, sigmas):

    """
    Average out degree of a homogeneous gaussian network
    """

    def compute_n_ij(net, sigma):

        sum = 0
        for x,y in zip(net.X[1:], net.Y[1:]):
            a = net.rho/(np.sqrt(2*np.pi)*sigma)
            k_ij = a*np.exp(-0.5*(x**2 + y**2)/(sigma**2))
            z_ij = 1 + k_ij**2
            sum += k_ij*(1-k_ij)/z_ij

        return sum

    avg_n_ij = np.zeros_like(sigmas)
    for i, sigma in enumerate(sigmas):
        print(sigma*np.sqrt(2*np.pi)*np.exp(1/(2*sigma**2)))
        avg_n_ij[i] = compute_n_ij(net, sigma)
    return avg_n_ij
