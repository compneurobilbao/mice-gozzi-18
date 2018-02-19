# -*- coding: utf-8 -*-
from src.env import DATA
from analysis.fig1_fig2_and_stats import plot_matrix, multipage
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
        all_fc_matrices[:,:,i] = np.load(opj(DATA, 'fc_matrices', file))
    
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
    from itertools import product         
    
    SOURCES = ['EL_filtered', 'EL_delta', 'EL_theta', 'EL_alpha', 'EL_beta',
               'EL_gamma', 'EL_gamma_high'] #, 'SC_BIN']
    TARGETS = ['FC_POS']
    ALPHA = 0.45
    BETA = 0.0
    MAX_CLUSTERS = 50
    output_dir = opj(CWD, 'reports', 'figures', 'active_state')
    
    figures = []

    for sub in SUBJECTS:
        input_dir_path = opj(CWD, 'reports', 'matrices', sub)
        legend = []
        for source, target in product(SOURCES, TARGETS):
            
            source_network =  np.load(opj(input_dir_path, source + '.npy'))
            target_network = np.load(opj(input_dir_path, target + '.npy'))
            legend.append(source + ' -> ' + target)

            result = np.zeros(MAX_CLUSTERS)
            
            for num_clusters in range(2,MAX_CLUSTERS):
                """
                Source dendogram -> target follows source
                """
                
                    
                if source in ['SC_BIN']: 
                    # SC_BIN discarded for the moment. 
                    # TODO: Calculation of Y
                    Z = cluster.hierarchy.linkage(Y, method='average')
                    T = cluster.hierarchy.cut_tree(Z,  n_clusters=num_clusters)
                    Xsf, Qff, Qsf, Lsf = cross_modularity(target_network,
                                                          source_network,
                                                          ALPHA,
                                                          BETA,
                                                          T[:, 0])
                    result[num_clusters] = np.nan_to_num(Xsf)
                else:
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
            plt.hold(True)
        plt.legend(legend)
        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
        plt.xlabel('# clusters')
        plt.ylabel('modularity value')
        plt.ylim((0, 0.5))
        ax = plt.title('Modularity_' + sub )
        fig = ax.get_figure()
        figures.append(fig)
        plt.close()
    
    multipage(opj(output_dir,
                  'xmod_el_2_fmri_pos.pdf'),
                    figures,
                    dpi=250)
    
        
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
  