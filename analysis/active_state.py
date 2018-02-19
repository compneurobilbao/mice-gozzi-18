# -*- coding: utf-8 -*-
from src.env import DATA, ATLAS

from analysis.bha import cross_modularity

import os
from os.path import join as opj
import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt
import nibabel as nib
from nilearn.input_data import NiftiLabelsMasker


CWD = os.getcwd()

# TR CALC
#func_img = nib.load('/home/asier/Downloads/final_folder/03_data/01_registered/ag150520a_10_veh_pcp_trm_partA_500_registered.nii.gz')
#func_img.header.get_zooms()
# TR = 1


def create_FC_matrices():
    from nilearn.connectome import ConnectivityMeasure
    
    for sub in os.listdir(opj(DATA, '03_data', '01_registered')):   
        
        
        
        output_dir_path = opj(DATA, 'fc_matrices')
        if not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path)
            
            preproc_data = opj(DATA, '03_data', '01_registered', file)

            atlas_path = ATLAS

            # atlas_2514
            masker = NiftiLabelsMasker(labels_img=atlas_path,
                                       background_label=0, verbose=5,
                                       detrend=True, standardize=True,
                                       t_r=1, # TR should not be a variable
                                       low_pass=0.1, high_pass=0.01)

            time_series = masker.fit_transform(preproc_data)


            # Save time series
            np.savetxt(opj(base_path, 'time_series_' + atlas + '.txt'),
                       time_series)






            
        # load function (conn matrix?)
        func_file = opj(DATA, 
        func_mat = np.loadtxt(func_file)

        correlation_measure = ConnectivityMeasure(kind='correlation')
        fc_mat = correlation_measure.fit_transform([func_mat])[0]
        
        fc_mat_neg = fc_mat.copy()
        fc_mat_pos = fc_mat.copy()
        
        fc_mat_neg[np.where(fc_mat>0)] = 0
        fc_mat_pos[np.where(fc_mat<0)] = 0


        np.save(opj(output_dir_path, 'FC.npy'),
                fc_mat) 

         
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
    
        
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
  