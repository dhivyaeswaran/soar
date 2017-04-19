import scipy
import time
from scipy import io

"""
This file is an example dataset loader file for YouTube. Create one file per
dataset in this directory, containing the load_data function.
The output format should be as given below.
"""

def load_data():
    """ Loads YouTube data of users and group memberships
    Args:
        None
    Returns:
        (dict) : keys - 'A' (adjacency matrix) or 'F' (membership matrix)
                 values - type scipy.sparse.csc.csc_matrix
    Note:
    * A should be binary and symmetric
    * F should be binary and have same #rows as A
    """
    mat = scipy.io.loadmat('data/youtube.mat')
    print 'YouTube: (#nodes, #edges, #attributes, #memberships):', \
    mat['F'].shape[0], mat['A'].nnz, mat['F'].shape[1], mat['F'].nnz

    # symmetrize adjacency matrix by adding reciprocal edges
    mat['A'] = mat['A'].maximum(mat['A'].T)

    return {'A': mat['A'].astype(int), 'F': mat['F'].astype(int)}


if __name__ == '__main__':
    start = time.time()
    data = load_data()
    end = time.time()
    print 'Loaded Youtube data of users and groups in', end - start, 'seconds'
