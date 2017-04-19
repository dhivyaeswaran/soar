import numpy as np
import scipy
from scipy import sparse
import matplotlib
import matplotlib.pyplot as plt
import cPickle as pickle
import time
import math

def sparse_matrix(edges, shape, symmetric=True):
    if symmetric:
        edges = edges[edges[:,0]!=edges[:,1],:]
        edges = np.vstack([ edges, edges[:,[1,0]] ])
    A = scipy.sparse.csc_matrix(
    (np.ones(edges.shape[0]), (edges[:,0], edges[:,1])), shape=shape
    )
    return scipy.sparse.csc_matrix((np.ones(A.nnz), A.nonzero()), shape=shape)

def sample_entries(Pi, m):
    nrow, ncol = Pi.shape
    probs = np.reshape(Pi, (1,-1))[0]/Pi.sum() # row by row
    idx = np.random.choice(len(probs), p=probs, size=(m,1))
    return np.hstack([idx/ncol, idx%ncol])

def sample_edges(P, m):
    nrow, ncol = P[0].shape[0], P[0].shape[1]
    edges = np.zeros((m,2))
    mul = np.repeat([[nrow, ncol]], m, axis=0)
    for i in xrange(len(P)):
        print '\t\tStep', i+1, 'of', len(P), '..'
        edges = edges*mul
        edges += sample_entries(P[i], m)
    return edges

def sample_graph(P0, M, noise=0, symmetric=True):
    if P0 is None:
        return None
    m = int(math.ceil(np.power(P0.sum(), M)))
    P = [P0 * (1 + noise*(np.random.random(P0.shape) - 0.5)) for t in xrange(M)]
    edges = sample_edges(P, m)
    shape = (np.power(P0.shape[0], M), np.power(P0.shape[1], M))
    print '\tCreating sparse matrix..'
    A = sparse_matrix(edges, shape, symmetric)
    return A

def mosaicg_generator(A0, F0, M, noise=0.4):
    """ MOSAIC-G generator for mosaicked graphs,
    i.e., graphs with many binary attributes; e.g., social-affiliation networks
    Args:
        A0 (np.array)    : initiator matrix for adjacency
        F0 (np.array)    : initiator matrix for membership
        M (int) : number of recursive steps for MOSAIC-G
    """
    print 'MOSAIC-G GENERATOR'
    start = time.time()
    print '\tAdjacency matrix..'
    A = sample_graph(A0, M, noise)
    print '\tMembership matrix..'
    F = sample_graph(F0, M, noise, symmetric=False)
    end = time.time()
    print '\tGenerated mosaicked graph in', end-start, 'seconds'
    print '\tPickling graph for later at results/mosaic-g/..'
    scipy.sparse.save_npz('results/mosaic-g/A.npz', A, compressed=True)
    scipy.sparse.save_npz('results/mosaic-g/F.npz', F, compressed=True)
    return {'A': A, 'F': F}
