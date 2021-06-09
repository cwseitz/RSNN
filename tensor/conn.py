import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from networkx.generators.random_graphs import erdos_renyi_graph

class InputWeights():

    def __init__(self,n_in,n_rec,p=0.1):

        self.p = p
        self.inputs, self.units = n_in, n_rec

        self.conn = self.conn()
        self.draw = self.draw_weights()
        self.weights = np.multiply(self.conn, self.draw)

    def conn(self, dtype=np.float32):
        _conn = np.zeros((self.inputs, self.units))
        for i in range(self.inputs):
            _conn[i] = np.random.choice([0,1], p=[1-self.p, self.p], size=(self.units,))
        _conn = _conn.astype(dtype)
        return _conn

    def draw_weights(self, wmax=0.1, dtype=np.float32):

        weights = np.zeros_like(self.conn)
        nonzero = np.argwhere(self.conn > 0)
        nonzero_count = len(nonzero)

        draw = np.random.lognormal(-1, 1, size=(nonzero_count,)).astype(dtype)

        for n in range(nonzero_count):
            x = nonzero[n][0]; y = nonzero[n][1]
            weights[x,y] = draw[n]

        return weights

class RecurrentWeights():

    def __init__(self, n_rec, p_e=1.0, pvec=[0.2, 0.3, 0.35, 0.25],
                 ex_mu=0,ex_sigma=1,in_mu=0,in_sigma=1, zero=False):

        self.p_ee, self.p_ii, self.p_ei, self.p_ie = pvec
        self.p_e = p_e; self.p_i = 1-self.p_e
        self.n_rec = n_rec
        self.n_excite = int(round(self.n_rec*self.p_e))
        self.n_inhib = int(round(self.n_rec*self.p_i))

        self.ex_mu = ex_mu
        self.ex_sigma = ex_sigma
        self.in_mu = in_mu
        self.in_sigma = in_sigma

        self.conn = self.conn()
        self.weights = self.conn
        self.draw = self.draw_weights()
        if zero:
            self.draw = np.zeros_like(self.conn)
        self.weights = np.multiply(self.conn, self.draw)


    def conn(self, dtype=np.float32):

        #Adjust probabilities for removing synaptic loops (directed graph)
        self.p_ee += self.p_ee**2
        self.p_ii += self.p_ii**2
        self.p_ei += self.p_ei*self.p_ie
        self.p_ie += self.p_ei*self.p_ie

        _conn = np.zeros((self.n_rec,self.n_rec))

        #Excitatory partition
        _conn[:round(self.n_excite), :round(self.n_excite)] = \
        np.random.binomial(1, p=self.p_ee, size=(round(self.n_excite),round(self.n_excite)))

        #Excitatory-inhibitory partition
        _conn[:round(self.n_excite), round(self.n_excite):] = \
        np.random.binomial(1, p=self.p_ei, size=(round(self.n_excite),round(self.n_inhib)))

        #Inhibitory-excitatory partition
        _conn[round(self.n_excite):, :round(self.n_excite)] = \
        np.random.binomial(1, p=self.p_ie, size=(round(self.n_inhib),round(self.n_excite)))

        #Inhibitory partition
        _conn[round(self.n_excite):, round(self.n_excite):] = \
        np.random.binomial(1, p=self.p_ii, size=(round(self.n_inhib),round(self.n_inhib)))
        _conn = _conn.astype(dtype)

        #Remove self connections
        np.fill_diagonal(_conn, 0)

        #Remove synaptic loops
        overlap = (_conn == _conn.T) * _conn
        _conn[np.nonzero(overlap)] = 0

        return _conn

    def draw_weights(self, dtype=np.float32):

        draw= np.zeros_like(self.conn)

        #Excitatory and excitatory-inhibitory partitions
        self.ex_shape = self.conn[:round(self.n_excite), :].shape
        ex_draw = np.random.lognormal(self.ex_mu, self.ex_sigma, size=self.ex_shape)
        ex_draw = ex_draw.astype(dtype)
        draw[:round(self.n_excite), :] = ex_draw

        #Inhibitory and inhibitory-excitatory partitions
        self.in_shape = self.conn[round(self.n_excite):, :].shape
        inh_draw = -np.random.lognormal(self.in_mu, self.in_sigma, size=self.in_shape)
        inh_draw = inh_draw.astype(dtype)
        draw[round(self.n_excite):, :] = inh_draw

        return draw

# def make_rec_weighted(conn, mu=1.13, sigma=0.5, dtype=np.float32):
#     weights = np.random.lognormal(mu, sigma, size=conn.shape).astype(dtype)
#     weighted = np.multiply(conn, weights)
#     return weighted

# def gen_rec_conn(units, p, dtype=np.float32):
#
#     g = erdos_renyi_graph(units, p, directed=True)
#     rec_conn = nx.to_numpy_matrix(g)
#     rec_conn = rec_conn.astype(dtype)
#
#     return rec_conn
