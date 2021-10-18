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

def dist(r_i, r_j, M, delta):
    x1, y1 = r_i; x2, y2 = r_j
    dx = np.minimum(np.abs(x1-x2),M*delta-np.abs(x1-x2))
    dy = np.minimum(np.abs(y1-y2),M*delta-np.abs(y1-y2))
    dr = np.sqrt(dx**2 + dy**2)
    return dr

def hogn_avg_out_deg(N, sigma, rho, delta, x0=0, y0=0):

    """
    Average out degree of a homogeneous gaussian network
    """

    M = int(round(np.sqrt(N)))
    grid = np.zeros((M,M))

    for x in range(M):
        for y in range(M):
            if x != x0 or y != y0:
                a = rho/(np.sqrt(2*np.pi)*sigma)
                dr_ij = dist((x0,y0), (x,y), M, delta)
                k_ij = a*np.exp(-0.5*(dr_ij**2)/(sigma**2))
                z_ij = 2*k_ij + (1-k_ij)**2
                grid[x,y] = k_ij/z_ij

    return grid

def hogn_var_out_deg(N, sigma, rho, delta, x0=0, y0=0):

    """
    Variance in the out degree of a homogeneous gaussian network
    """

    M = int(round(np.sqrt(N)))
    grid = np.zeros((M,M))

    for x in range(M):
        for y in range(M):
            if x != x0 or y != y0:
                a = rho/(np.sqrt(2*np.pi)*sigma)
                dr_ij = dist((x0,y0), (x,y), M, delta)
                k_ij = a*np.exp(-0.5*(dr_ij**2)/(sigma**2))
                z_ij = 2*k_ij + (1-k_ij)**2
                p_ij = k_ij/z_ij
                grid[x,y] = p_ij*(1-p_ij)

    return grid
