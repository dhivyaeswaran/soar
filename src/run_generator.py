import os
import numpy as np
from src.generator.mosaicg import mosaicg_generator
from src.patterns import distributions, pointvalues

if __name__=='__main__':
    os.system('mkdir -p results')
    os.system('mkdir -p results/mosaic-g')
    l, o = 0.62, 1e-4 # 1 and 0 respectively
    A0 = np.array([
        [l,l,l,l,o],
        [l,l,l,o,o],
        [l,l,l,o,o],
        [l,o,o,l,o],
        [o,o,o,o,l/4]
    ])
    F0 = np.array([
    [l,l,l,l],
    [l,l,o,o],
    [l,o,l,o],
    [l,o,o,o],
    [l/4,o,o,o]
    ])
    M = 8
    noise = 0.5
    data = mosaicg_generator(A0, F0, M, noise)
    print 'MOSAIC-G: (#nodes, #edges, #attributes, #memberships):', \
    data['F'].shape[0], data['A'].nnz, data['F'].shape[1], data['F'].nnz
    distributions.get_patterns('mosaic-g', data)
    pointvalues.get_patterns('mosaic-g', data)
    print 'Check results/mosaic-g directory for results..'
