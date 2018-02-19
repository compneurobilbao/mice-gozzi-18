from src.dmri import (run_spm_fsl_dti_preprocessing,
                      run_dti_artifact_correction,
                      correct_dwi_space_atlas,
                      get_con_matrix_matlab,
                      run_dtk_tractography,
                      )

from src.interfaces import (CanICAInterface,
                            plot_all_components,
                            plot_ica_components,
                            plot_multi_slices,
                            plot_overlays,
                            plot_stat_overlay,
                            KellyKapowski,
                            EddyCorrect,
                            Eddy,
                            )

from src.utils import (spm_tpm_priors_path,
                       remove_ext,
                       rename,
                       get_extension,
                       get_affine,
                       get_data_dims,
                       get_vox_dims,
                       fetch_one_file,
                       extension_duplicates,
                       extend_trait_list,
                       selectindex,
                       get_trait_value,
                       fsl_merge,
                       get_node,
                       joinstrings,
                       find_wf_node,
                       get_datasink,
                       get_input_node,
                       get_interface_node,
                       get_input_file_name,
                       add_table_headers,
                       write_tabbed_excel,
                       )

from src.preproc import (PETPVC,
                         nipy_motion_correction,
                         nlmeans_denoise_img,
                         create_regressors,
                         extract_noise_components,
                         motion_regressors,
                         reslice_img,
                         spm_apply_deformations,
                         spm_coregister,
                         spm_normalize,
                         spm_warp_to_mni,
                         afni_deoblique,
                         spm_tpm_priors_path,
                         spm_create_group_template_wf,
                         spm_register_to_template_wf,
                         afni_slicetime,
                         spm_slicetime,
                         auto_spm_slicetime,
                         auto_nipy_slicetime,
                         STCParameters,
                         STCParametersInterface,
                         get_bounding_box,
                         run_fmriprep,
                         run_mriqc,
                         )

from src.data import dcm2bids