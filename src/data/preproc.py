# -*- coding: utf-8 -*-
from src.env import DATA, ATLAS

import os
from os.path import join as opj
import numpy as np
import nibabel as nib
from nilearn.input_data import NiftiLabelsMasker

CWD = os.getcwd()

# TR CALC
# func_img = nib.load('/home/asier/Downloads/final_folder/03_data/01_registered/ag150520a_10_veh_pcp_trm_partA_500_registered.nii.gz')
# func_img.header.get_zooms()
# TR = 1


def atlas_renaming():

    atlas_img = nib.load('/home/asier/Downloads/final_folder/02_atlases_and_masks/01_atlas_masked_no_cereb_no_ventr.nii.gz')
    fmri_img = nib.load('/home/asier/Downloads/final_folder/03_data/01_registered/ag150520a_10_veh_pcp_trm_partA_500_registered.nii.gz')
    atlas_data = atlas_img.get_data()

    atlas_new_data = np.zeros((atlas_data.shape), dtype='int')

    # left hemisphere
    for number, roi in enumerate(range(1000, 1127)):
        atlas_new_data[np.where(atlas_data == roi)] = number+1

    # right hemisphere
    for number, roi in enumerate(range(100, 227)):
        atlas_new_data[np.where(atlas_data == roi)] = number+1+127

    atlas_new_data_img = nib.Nifti1Image(atlas_new_data,
                                         affine=fmri_img.affine)

    nib.save(atlas_new_data_img,
             '/home/asier/Downloads/final_folder/02_atlases_and_masks/atlas_corrected_1_255_rois.nii.gz')


def create_FC_matrices():
    from nilearn.connectome import ConnectivityMeasure

    for sub in os.listdir(opj(DATA, '03_data', '01_registered')):
        print('Processing subject: ' + sub)

        mat_output_dir_path = opj(DATA, 'fc_matrices')
        if not os.path.exists(mat_output_dir_path):
            os.makedirs(mat_output_dir_path)
        signal_output_dir_path = opj(DATA, 'signals')
        if not os.path.exists(signal_output_dir_path):
            os.makedirs(signal_output_dir_path)

        preproc_data = opj(DATA, '03_data', '01_registered', sub)

        atlas_path = ATLAS

        masker = NiftiLabelsMasker(labels_img=atlas_path,
                                   background_label=0, verbose=5,
                                   detrend=True, standardize=True,
                                   t_r=1,  # TR should not be a variable
                                   low_pass=0.1, high_pass=0.01)

        time_series = masker.fit_transform(preproc_data)

        # Save time series
        np.savetxt(opj(signal_output_dir_path, 'time_series_' + sub + '.txt'),
                   time_series)

        correlation_measure = ConnectivityMeasure(kind='correlation')
        fc_mat = correlation_measure.fit_transform([time_series])[0]

        np.save(opj(mat_output_dir_path, 'FC_' + sub + '.npy'),
                fc_mat)
