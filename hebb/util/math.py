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
