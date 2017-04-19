import scipy
from scipy import sparse, stats
from scipy.sparse import linalg as la
from scipy.sparse import csgraph
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cPickle as pickle
from collections import Counter
import sys
from src.settings import SIZES

################################################################################
#                              COMPUTE  FUNCTIONS                              #
################################################################################

def get_patterns(dataset_name, data, v_min=2):
    """
    Args:
        dataset_name (str) : name of dataset, determines where result is stored
        data (dict<str,scipy.sparse.csc.csc_matrix>) : loaded data of adjacency
            and membership matrices
        v_min (int): minimum volume cutoff for AIS
    Outputs:
        (dict(dict)) : dictionary of attribute induced subgraph properties
    """
    print 'POINT_VALUE PATTERNS'
    A, F = data['A'], data['F']
    # determine attributes for which AIS should be computed
    num_attrs = F.shape[1]
    attr_degrees = np.array(F.sum(axis=0))[0]
    attrs = [a for a in xrange(num_attrs) if attr_degrees[a] >= v_min]
    # compute AIS properties
    values = {}
    for i,a in enumerate(attrs):
        if i % 100 == 0:
            print '\tAIS', i, 'of', len(attrs), '..'
        Vsub = scipy.sparse.find(F[:,a])[0]
        Asub = A[Vsub,:][:, Vsub]
        values[a] = {
        'volume': len(Vsub),
        'mass': Asub.nnz / 2,
        'surface area': A[Vsub,:].sum() - Asub.nnz,
        'triangle count': (Asub.dot(Asub)).multiply(Asub).sum()/6,
        'spectral radius': np.max(la.svds(
        Asub.asfptype(), k=1, return_singular_vectors=False
        )),
        'GCC size': np.max(get_counts(
        csgraph.connected_components(Asub, connection='strong')[1]
        )[:,1])
        }
    # plot patterns
    plot(values, dataset_name)
    # save patterns for later
    with open('results/%s/pointvalues.pkl'%dataset_name, 'wb') as f:
        pickle.dump(values, f)

################################################################################
#                               PLOT  FUNCTIONS                                #
################################################################################

def plot(values, dataset_name, zero_lift=1):
    """
    Args:
        values (dict) : dictionary of distribution values
        dataset_name (str) : dataset's name to store results in
    """
    items = [
    ('volume', 'mass'),
    ('mass', 'surface area'),
    ('triangle count', 'mass'),
    ('triangle count', 'spectral radius'),
    ('GCC size', 'volume')
    ]
    for x_label, y_label in items:
        print '\tPlotting', y_label, 'vs', x_label
        fig = plt.figure()
        ax = fig.add_subplot(111)
        x, y = [], []
        for _, value in values.iteritems():
            x.append(value[x_label] + zero_lift)
            y.append(value[y_label] + zero_lift)
        x, y = np.array(x), np.array(y)
        line = log_fit_line(x, y)
        # plot
        ax.scatter(x, y, color='darkgray', alpha=0.2, marker='o')
        ax.scatter(line['x'], line['bin_avg'], color='k', marker='^', s=100)
        ax.plot(line['x'], line['y'], color='k', linestyle='--', linewidth=4)
        # title, label, ticks, scale
        ax.set_title(dataset_name.upper(), fontsize=SIZES['title'])
        ax.set_xlabel(x_label, fontsize=SIZES['label'])
        ax.set_ylabel(y_label, fontsize=SIZES['label'])
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim([0.6*np.min(x), np.max(x)*5])
        ax.set_ylim([0.6*np.min(y), np.max(y)*5])
        ax.xaxis.set_tick_params(labelsize=SIZES['tick'])
        ax.yaxis.set_tick_params(labelsize=SIZES['tick'])
        # annotation
        ax.text(0.95, 0.15, 'corr: %1.2f'%line['corr'],
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes, fontsize=SIZES['annotation'])
        ax.text(0.95, 0.03, 'slope: %1.2f'%line['m'],
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes, fontsize=SIZES['annotation'])
        # save
        plt.gcf().subplots_adjust(bottom=0.18, left=0.18)
        fname = 'results/%s/%s_vs_%s.png' % (dataset_name, y_label, x_label)
        fig.savefig(fname)
        plt.clf()

################################################################################
#                             UTILITY  FUNCTIONS                               #
################################################################################

def get_counts(v):
    """ Compute number of times each value occurs in the list.
    Args:
        v (array-like) : 1d array of values (usually, degrees)
    Returns:
        (np.array) : 2d array with each row denoting (value, its count)
    """
    return np.array(Counter(v).items())

def log_fit_line(x, y, nbins=20):
    """ Fits line by logarithmic binning of x values
    Args:
        x (array-list) : 1d array of positive values
        y (array-list) : 1d array of positive values
    Returns:
        (dict) : dictionary of bin_avg, line's x and y, slope, corr values
    """
    try:
        assert np.min(x) != np.max(x)
    except:
        'x is a constant, binning failed in log_fit_line()'
        sys.exit(1)
    x_edges =  np.logspace(
    base=2, start=np.log2(np.min(x)), stop=np.log2(np.max(x)), num=nbins+1
    )
    x_bins, y_bins = np.zeros(nbins), np.zeros(nbins)
    for i in xrange(nbins):
        x_bins[i] = np.average(x_edges[i:i+2])
        y_bins[i] = np.average(y[np.logical_and(x>=x_edges[i], x<=x_edges[i+1])])
    select_bins = y_bins>0
    select_bins[0] = False # ignore the first bin as it has zero counts
    line = np.polyfit(np.log(x_bins[select_bins]),np.log(y_bins[select_bins]), 1)
    return {
    'm': line[0],
    'x': x_bins[select_bins],
    'bin_avg': y_bins[select_bins],
    'y': np.exp(np.poly1d(line)(np.log(x_bins[select_bins]))),
    'corr': scipy.stats.pearsonr(
    np.log(x_bins[select_bins]), np.log(y_bins[select_bins])
    )[0]
    }
