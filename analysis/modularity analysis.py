# -*- coding: utf-8 -*-
from src.env import DATA
from analysis.fig1_fig2_and_stats import multipage, plot_matrix
from analysis.bha import cross_modularity

import os
from os.path import join as opj
import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt


CWD = os.getcwd()


def create_fc_mean_matrix():

    all_fc_matrices = np.zeros((254, 254, 30))

    for i, file in enumerate(os.listdir(opj(DATA, 'fc_matrices'))):
        print(file, i)
        all_fc_matrices[:, :, i] = np.load(opj(DATA, 'fc_matrices', file))

    mean_all_fc_matrices = np.mean(all_fc_matrices, axis=2)
    np.save(opj(DATA, 'FC_mean_all_subject.npy'),
            mean_all_fc_matrices)


def create_sc_mean_matrix():

    mat = sio.loadmat(opj(DATA, '03_data', 'partial_connectome_directed.mat'))
    sc_matrix = mat['partial_connectome_asymm']
    np.save(opj(DATA, 'SC_connectome.npy'),
            sc_matrix)


def modularity_analysis():

    from scipy import spatial, cluster

    ALPHA = 0.45
    BETA = 0.0
    MAX_CLUSTERS = 50
    output_dir = opj(CWD, 'reports')

    figures = []
    source_network = np.load(opj(DATA, 'SC_connectome.npy'))
    #source_network[np.where(source_network>0)] = 1
    target_network = np.load(opj(DATA, 'FC_mean_all_subject.npy'))
    target_network = np.abs(target_network)

    result = np.zeros(MAX_CLUSTERS)
    for num_clusters in range(2, MAX_CLUSTERS):
        """
        Source dendogram -> target follows source
        """
        Y = spatial.distance.pdist(source_network, metric='cosine')
        Y = np.nan_to_num(Y)
        Z = cluster.hierarchy.linkage(Y, method='weighted')
        T = cluster.hierarchy.cut_tree(Z, n_clusters=num_clusters)

        Xsf, Qff, Qsf, Lsf = cross_modularity(target_network,
                                              source_network,
                                              ALPHA,
                                              BETA,
                                              T[:, 0])
        result[num_clusters] = np.nan_to_num(Xsf)

    plt.plot(result)
    plt.xlabel('# clusters')
    plt.ylabel('modularity value')
    plt.ylim((0, 0.2))
    ax = plt.title('Modularity_sc_2_fc')
    fig = ax.get_figure()
    figures.append(fig)
    plt.close()

    multipage(opj(output_dir,
                  'xmod_sc_2_fmri.pdf'),
              figures,
              dpi=250)


def plotting_clusters():

    BEST_XMOD = 8
    ALPHA = 0.45
    BETA = 0.0
    output_dir = opj(CWD, 'reports')

    source_network = np.load(opj(DATA, 'SC_connectome.npy'))
    target_network = np.load(opj(DATA, 'FC_mean_all_subject.npy'))
    target_network = np.abs(target_network)

    """
    Source dendogram -> target follows source
    """
    Y = spatial.distance.pdist(source_network, metric='cosine')
    Y = np.nan_to_num(Y)
    Z = cluster.hierarchy.linkage(Y, method='weighted')
    T = cluster.hierarchy.cut_tree(Z, n_clusters=BEST_XMOD)

    Xsf, Qff, Qsf, Lsf = cross_modularity(target_network,
                                          source_network,
                                              ALPHA,
                                              BETA,
                                              T[:, 0])
    
    sort_idx = np.argsort(np.squeeze(T))
    reorder_source = source_network[sort_idx,:]
    plot_matrix(reorder_source, arange(255))
    
    reorder_target = target_network[sort_idx,:]
    plot_matrix(reorder_target, arange(255))
    
    plot_matrix(source_network, arange(255))
    
    
    
    
    
    
    
    