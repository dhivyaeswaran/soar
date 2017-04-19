import scipy
from scipy import sparse
from scipy.sparse import linalg as la
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import cPickle as pickle
from src.settings import SIZES

################################################################################
#                              COMPUTE  FUNCTIONS                              #
################################################################################

def get_patterns(dataset_name, data, k=100):
    """
    Args:
        dataset_name (str) : name of dataset, determines where result is stored
        data (dict<str,scipy.sparse.csc.csc_matrix>) : loaded data of adjacency
            and membership matrices
        k (int) : number of singular values to compute
    Outputs:
        None. Check results/<dataset_name> directory.
    """
    print 'DISTRIBUTION PATTERNS'
    A, F = data['A'], data['F']
    values = {}
    # number of nodes a node is connected to
    print '\tComputing node degree..'
    values['node degree'] = np.array(A.sum(axis=1))[:,0]
    # number of attributes a node has
    print '\tComputing attribute degree..'
    values['attribute degree'] = np.array(F.sum(axis=1))[:,0]
    # number of nodes in an attribute-induced subgraph
    print '\tComputing volume..'
    values['volume'] = np.array(F.sum(axis=0))[0,:] # num nodes in attind subgraph
    # number of edges in an attribute-induced subgraph
    print '\tComputing mass.. (may take a while)'
    AF = A.dot(F)
    values['mass'] = np.array( (AF.multiply(F)).sum(0) ).flatten() / 2
    # singular values of adjacency, membership and degree matrices
    print '\tComputing adjacency spectrum.. (may take a while)'
    values['A spectra'] = la.svds(
    A.asfptype(), k, return_singular_vectors=False
    )
    print '\tComputing membership spectrum.. (may take a while)'
    values['F spectra'] = la.svds(
    F.asfptype(), k, return_singular_vectors=False
    )
    print '\tComputing degree spectrum.. (may take a while)'
    values['AF spectra'] = la.svds(
    AF.asfptype(), k, return_singular_vectors=False
    )
    # plot patterns
    rank_plots(values, dataset_name)
    nonrank_plots(values, dataset_name)
    # save patterns for later
    print '\tPickling values for later..'
    with open('results/%s/distributions.pkl'%dataset_name, 'wb') as f:
            pickle.dump(values, f)

################################################################################
#                               PLOT  FUNCTIONS                                #
################################################################################

def nonrank_plots(values, dataset_name):
    """
    Args:
        values (dict) : dictionary of distribution values
        dataset_name (str) : dataset's name to store results in
    """
    items = {
    'node or attribute degree': ('node degree', 'attribute degree'),
    'mass or volume':  ('mass', 'volume')
    }
    for x_label, properties in items.iteritems():
        print '\tPlotting', x_label, 'distributions..'
        p0, p1 = properties[0], properties[1]
        dist0_raw = get_counts(values[p0])
        dist1_raw = get_counts(values[p1])
        for mode in ['frequency']:
            dist0 = convert_counts(dist0_raw, mode)
            dist1 = convert_counts(dist1_raw, mode)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(dist0[:,0], dist0[:,1], linewidth=2.0, color='b', label=p0)
            ax.plot(dist1[:,0], dist1[:,1], linewidth=2.0, color='r', label=p1)
            # axis scale/label/ticks
            ax.set_title(dataset_name.upper(), fontsize=SIZES['title'])
            ax.set_xlabel(x_label, fontsize=SIZES['label'])
            ax.set_ylabel(mode, fontsize=SIZES['label'])
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.xaxis.set_tick_params(labelsize=SIZES['tick'])
            ax.yaxis.set_tick_params(labelsize=SIZES['tick'])
            # legend
            legend = ax.legend(loc=4) if mode=='odds' else ax.legend(loc=1)
            for label in legend.get_texts():
                label.set_fontsize(SIZES['annotation'])
            # save
            plt.gcf().subplots_adjust(bottom=0.18, left=0.18)
            fig.savefig('results/%s/%s_%s.png'%(dataset_name, x_label, mode))
            plt.clf()

def rank_plots(values, dataset_name):
    """ Plots and saves rank plots (ie, singular values, for now)
    Args:
        values (dict)   : dictionary of distribution values
        dataset_name (str) : dataset's name to store results in
    """
    items = {
    'singular value': ('A spectra', 'F spectra', 'AF spectra')
    }
    for x_label, properties in items.iteritems():
        print '\tPlotting', x_label, 'distributions..'
        p0, p1, p2 = properties[0], properties[1], properties[2]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        rank = 1 + np.arange(len(values[p0]))
        ax.plot(
        rank, values[p0][::-1], linewidth=2.0, color='b',
        label='A (%.2f)' % log_slope(rank, values[p0])
        )
        ax.plot(
        rank, values[p1][::-1], linewidth=2.0, color='g',
        label='F (%.2f)' % log_slope(rank, values[p1])
        )
        ax.plot(
        rank, values[p2][::-1], linewidth=2.0, color='r',
        label='AF (%.2f)' % log_slope(rank, values[p2])
        )
        # axis scale/label/ticks
        ax.set_title(dataset_name.upper(), fontsize=SIZES['title'])
        ax.set_xlabel('rank', fontsize=SIZES['label'])
        ax.set_ylabel(x_label, fontsize=SIZES['label'])
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.xaxis.set_tick_params(labelsize=SIZES['tick'])
        ax.yaxis.set_tick_params(labelsize=SIZES['tick'])
        # legend
        legend = ax.legend()
        for label in legend.get_texts():
            label.set_fontsize(SIZES['annotation'])
        # save
        plt.gcf().subplots_adjust(bottom=0.18, left=0.18)
        fig.savefig('results/%s/%s_rank.png'%(dataset_name, x_label))
        plt.clf()

################################################################################
#                             UTILITY  FUNCTIONS                               #
################################################################################
def convert_counts(raw, mode):
    ind = np.argsort(raw[:,0])
    if mode == 'frequency':
        new = np.zeros(raw.shape)
        new[:,0] = raw[ind,0]
        new[:,1] = raw[ind,1]
    if mode == 'ccdf':
        new = np.zeros(raw.shape)
        new[:,0] = raw[ind,0]
        pdf = raw[ind,1]/float(np.sum(raw[:,1]))
        new[:,1] = pdf[::-1].cumsum()[::-1]
    if mode == 'odds':
        new = np.zeros((raw.shape[0]-1, 2))
        new[:,0] = raw[ind[:-1],0]
        pdf = raw[ind,1]/float(np.sum(raw[:,1]))
        cdf = pdf.cumsum()
        ccdf = 1-cdf
        new[:,1] = cdf[:-1]/ccdf[:-1]
    return new

def get_counts(v):
    """ Compute number of times each value occurs in the list, OMITTING ZEROS.
    Args:
        v (array-like) : 1d array of values (usually, degrees)
    Returns:
        (np.array) : 2d array with each row denoting (value, its count)
    """
    return np.array(Counter(v[np.nonzero(v)]).items())

def log_slope(x, y):
    """ Computes slope in log scale (assuming values are positive)
    Args:
        x (array-list) : 1d array of positive values
        y (array-list) : 1d array of positive values
    Returns:
        (float) : slope of the line fitted to x and y in log scale
    """
    return np.polyfit(np.log(x), np.log(y), 1)[0]
