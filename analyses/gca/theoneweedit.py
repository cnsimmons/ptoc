#GCA Searchlight

import sys
import os
import logging
import pandas as pd
import numpy as np
from nilearn import image, maskers
from nilearn.glm.first_level import compute_regressor
from statsmodels.tsa.stattools import grangercausalitytests
import nibabel as nib
from tqdm import tqdm
from brainiak.searchlight.searchlight import Ball
from mpi4py import MPI
from nilearn.input_data import NiftiMasker
from nilearn.decoding import SearchLight

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import your parameters
curr_dir = f'/user_data/csimmon2/git_repos/ptoc'
sys.path.insert(0, curr_dir)
import ptoc_params as params

# Set up directories and parameters
study = 'ptoc'
study_dir = f"/lab_data/behrmannlab/vlad/{study}"
localizer = 'Object'
results_dir = '/user_data/csimmon2/git_repos/ptoc/results'
raw_dir = params.raw_dir

# Load subject information
sub_info = pd.read_csv(f'{curr_dir}/sub_info.csv')
sub_info = sub_info[sub_info['group'] == 'control']
subs = ['sub-025']
rois = ['pIPS']
hemispheres = ['left']

run_num = 3
runs = list(range(1, run_num + 1))
run_combos = [[rn1, rn2] for rn1 in range(1, run_num + 1) for rn2 in range(rn1 + 1, run_num + 1)]

def extract_roi_sphere(img, coords):
    roi_masker = maskers.NiftiSpheresMasker([tuple(coords)], radius=6)
    seed_time_series = roi_masker.fit_transform(img)
    phys = np.mean(seed_time_series, axis=1).reshape(-1, 1)
    return phys

def make_psy_cov(runs, ss):
    temp_dir = f'{raw_dir}/{ss}/ses-01'
    cov_dir = f'{temp_dir}/covs'
    vols_per_run, tr = 184, 2.0
    total_vols = vols_per_run * len(runs)
    times = np.arange(0, total_vols * tr, tr)
    full_cov = pd.DataFrame(columns=['onset', 'duration', 'value'])

    for i, rn in enumerate(runs):
        ss_num = ss.split('-')[1]
        obj_cov_file = f'{cov_dir}/catloc_{ss_num}_run-0{rn}_{localizer}.txt'

        if not os.path.exists(obj_cov_file):
            logging.warning(f'Covariate file not found for run {rn}')
            continue

        obj_cov = pd.read_csv(obj_cov_file, sep='\t', header=None, names=['onset', 'duration', 'value'])
        
        if i > 0:
            obj_cov['onset'] += i * vols_per_run * tr
        
        full_cov = pd.concat([full_cov, obj_cov])

    full_cov = full_cov.sort_values(by=['onset']).reset_index(drop=True)
    cov = full_cov.to_numpy()
    valid_onsets = cov[:, 0] < times[-1]
    cov = cov[valid_onsets]

    if cov.shape[0] == 0:
        logging.warning('No valid covariate data after filtering. Returning zeros array.')
        return np.zeros((total_vols, 1))

    psy, _ = compute_regressor(cov.T, 'spm', times)
    psy[psy > 0] = 1
    psy[psy <= 0] = 0
    return psy

def extract_cond_ts(ts, cov):
    block_ind = (cov==1)
    block_ind = np.insert(block_ind, 0, True)
    block_ind = np.delete(block_ind, len(block_ind)-1)
    block_ind = (cov == 1).reshape((len(cov))) | block_ind
    return ts[block_ind]

def conduct_gca():
    logging.info(f'Running GCA for {localizer} with searchlight...')
    tasks = ['loc']
    
    for ss in subs:
        sub_summary = pd.DataFrame(columns=['sub', 'fold', 'task', 'origin', 'target_x', 'target_y', 'target_z', 'f_diff'])
        
        sub_dir = f'{study_dir}/{ss}/ses-01/'
        temp_dir = f'{raw_dir}/{ss}/ses-01'
        roi_dir = f'{sub_dir}/derivatives/rois'
        exp_dir = f'{temp_dir}/derivatives/fsl/loc'
        os.makedirs(f'{sub_dir}/derivatives/gca', exist_ok=True)

        roi_coords = pd.read_csv(f'{roi_dir}/spheres/sphere_coords_hemisphere.csv')
        logging.info(f"ROI coordinates loaded for subject {ss}")

        for rcn, rc in enumerate(run_combos):
            logging.info(f"Processing run combination {rc} for subject {ss}")
            
            filtered_list = []
            brain_masks = []
            for rn in rc:
                curr_run = image.load_img(f'{exp_dir}/run-0{rn}/1stLevel.feat/filtered_func_data_reg.nii.gz')
                curr_run = image.clean_img(curr_run, standardize=True)
                filtered_list.append(curr_run)
                
                # Load brain mask for each run
                brain_mask = image.load_img(f'{exp_dir}/run-0{rn}/1stLevel.feat/mask.nii.gz')
                brain_masks.append(brain_mask)

            img4d = image.concat_imgs(filtered_list)
            logging.info(f"Concatenated image shape: {img4d.shape}")

            # Create combined brain mask
            if len(brain_masks) > 1:
                combined_mask_data = np.all([mask.get_fdata() > 0 for mask in brain_masks], axis=0)
                combined_brain_mask = nib.Nifti1Image(combined_mask_data.astype(np.int32), brain_masks[0].affine)
            else:
                combined_brain_mask = brain_masks[0]

            # Create searchlight object
            searchlight = SearchLight(
                combined_brain_mask, 
                process_mask_img=combined_brain_mask, 
                radius=6,
                n_jobs=-1,  # Use all available cores
                verbose=0
            )

            for tsk in tasks:
                for dorsal_roi in rois:
                    for dorsal_hemi in hemispheres:
                        dorsal_coords = roi_coords[(roi_coords['index'] == rcn) & 
                                                   (roi_coords['task'] == tsk) & 
                                                   (roi_coords['roi'] == dorsal_roi) &
                                                   (roi_coords['hemisphere'] == dorsal_hemi)]
                        
                        if dorsal_coords.empty:
                            logging.warning(f"No coordinates found for {dorsal_roi}, {dorsal_hemi}, run combo {rc}")
                            continue

                        dorsal_ts = extract_roi_sphere(img4d, dorsal_coords[['x', 'y', 'z']].values.tolist()[0])
                        
                        psy = make_psy_cov(rc, ss)
                        
                        if dorsal_ts.shape[0] != psy.shape[0]:
                            raise ValueError(f"Mismatch in volumes: dorsal_ts has {dorsal_ts.shape[0]}, psy has {psy.shape[0]}")
                        
                        dorsal_phys = extract_cond_ts(dorsal_ts, psy)

                        # Function to compute GCA for each searchlight sphere
                        def compute_gca(sphere_signals):
                            sphere_ts = np.mean(sphere_signals, axis=1)
                            sphere_phys = extract_cond_ts(sphere_ts.reshape(-1, 1), psy)

                            neural_ts = pd.DataFrame({
                                'dorsal': dorsal_phys.ravel(), 
                                'sphere': sphere_phys.ravel()
                            })
                            
                            gc_res_dorsal = grangercausalitytests(neural_ts[['sphere', 'dorsal']], 1, verbose=False)
                            gc_res_sphere = grangercausalitytests(neural_ts[['dorsal', 'sphere']], 1, verbose=False)

                            return gc_res_dorsal[1][0]['ssr_ftest'][0] - gc_res_sphere[1][0]['ssr_ftest'][0]

                        # Run searchlight analysis
                        f_diff_map = searchlight.fit(img4d, compute_gca)

                        # Save the f_diff_map
                        f_diff_img = nib.Nifti1Image(f_diff_map, img4d.affine)
                        nib.save(f_diff_img, f'{sub_dir}/derivatives/gca/f_diff_map_{localizer.lower()}_{dorsal_roi}_{dorsal_hemi}_rc{rcn}.nii.gz')

                        # Summarize results
                        dorsal_label = f"{dorsal_hemi[0]}{dorsal_roi}"
                        for x in range(f_diff_map.shape[0]):
                            for y in range(f_diff_map.shape[1]):
                                for z in range(f_diff_map.shape[2]):
                                    if combined_brain_mask.get_fdata()[x, y, z] > 0:
                                        curr_data = pd.Series([ss, rcn, tsk, dorsal_label, x, y, z, f_diff_map[x, y, z]], 
                                                              index=sub_summary.columns)
                                        sub_summary = sub_summary.append(curr_data, ignore_index=True)
                        
                        logging.info(f"Completed GCA for {ss}, {tsk}, {dorsal_label}, searchlight")

        logging.info(f'Completed GCA for subject {ss}')
        sub_summary.to_csv(f'{sub_dir}/derivatives/gca/gca_summary_{localizer.lower()}_searchlight.csv', index=False)

    return sub_summary

if __name__ == "__main__":
    conduct_gca()