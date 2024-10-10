import sys
import os
import logging
import pandas as pd
import numpy as np
from nilearn import image, input_data
from nilearn.input_data import NiftiMasker
from nilearn.glm.first_level import compute_regressor
from nilearn.decoding import SearchLight
from statsmodels.tsa.stattools import grangercausalitytests
import nibabel as nib
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import your parameters
curr_dir = f'/user_data/csimmon2/git_repos/ptoc'
sys.path.insert(0, curr_dir)
import ptoc_params as params

# Set up directories and parameters
study = 'ptoc'
study_dir = f"/lab_data/behrmannlab/vlad/{study}"
localizer = 'Object'  # scramble or object. This is the localizer task.
results_dir = '/user_data/csimmon2/git_repos/ptoc/results'
raw_dir = params.raw_dir

# Load subject information
sub_info = pd.read_csv(f'{curr_dir}/sub_info.csv')
sub_info = sub_info[sub_info['group'] == 'control']
subs = sub_info['sub'].tolist()

rois = ['pIPS', 'LO']
hemispheres = ['left', 'right']
run_num = 3
runs = list(range(1, run_num + 1))
run_combos = [[rn1, rn2] for rn1 in range(1, run_num + 1) for rn2 in range(rn1 + 1, run_num + 1)]

def extract_roi_sphere(img, coords, radius=6):
    if not isinstance(coords, (list, tuple)):
        coords = [coords]
    coords = [tuple(coord) if isinstance(coord, (list, np.ndarray)) else (coord,) for coord in coords]
    
    roi_masker = input_data.NiftiSpheresMasker(coords, radius=radius)
    seed_time_series = roi_masker.fit_transform(img)
    return np.mean(seed_time_series, axis=1).reshape(-1, 1)

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

def gca_calculation(sphere_ts, roi_ts, psy):
    sphere_phys = extract_cond_ts(sphere_ts, psy)
    roi_phys = extract_cond_ts(roi_ts, psy)
    
    try:
        neural_ts = pd.DataFrame({'sphere': sphere_phys.ravel(), 'roi': roi_phys.ravel()})
        gc_res_sphere_to_roi = grangercausalitytests(neural_ts[['sphere', 'roi']], 1, verbose=False)
        gc_res_roi_to_sphere = grangercausalitytests(neural_ts[['roi', 'sphere']], 1, verbose=False)
        f_diff = gc_res_sphere_to_roi[1][0]['ssr_ftest'][0] - gc_res_roi_to_sphere[1][0]['ssr_ftest'][0]
    except Exception as e:
        logging.warning(f"Error in GCA calculation: {str(e)}")
        f_diff = 0
    
    return f_diff

def conduct_gca_searchlight():
    logging.info(f'Running GCA searchlight for {localizer}...')
    
    for ss in subs:
        sub_summary = pd.DataFrame(columns=['sub', 'roi', 'x', 'y', 'z', 'task', 'f_diff'])
        
        sub_dir = f'{study_dir}/{ss}/ses-01/'
        temp_dir = f'{raw_dir}/{ss}/ses-01'
        roi_dir = f'{sub_dir}/derivatives/rois'
        exp_dir = f'{temp_dir}/derivatives/fsl/loc'
        os.makedirs(f'{sub_dir}/derivatives/gca_searchlight', exist_ok=True)

        roi_coords = pd.read_csv(f'{roi_dir}/spheres/sphere_coords_hemisphere.csv')
        logging.info(f"ROI coordinates loaded for subject {ss}")

        for rcn, rc in enumerate(run_combos):
            logging.info(f"Processing run combination {rc} for subject {ss}")
            
            filtered_list = []
            for rn in rc:
                brain_mask_path = f'{sub_dir}/derivatives/fsl/loc/run-0{rn}/1stLevel.feat/mask.nii.gz'
                brain_mask = nib.load(brain_mask_path)
            
                curr_run = image.load_img(f'{exp_dir}/run-0{rn}/1stLevel.feat/filtered_func_data_reg.nii.gz')
                curr_run = image.clean_img(curr_run, standardize=True)
                filtered_list.append(curr_run)
                
            img4d = image.concat_imgs(filtered_list)
            logging.info(f"Concatenated image shape: {img4d.shape}")
            
            # Create psychological covariate
            psy = make_psy_cov(rc, ss)

            # Extract ROI time series (e.g., pIPS)
            roi_coord = roi_coords[(roi_coords['index'] == rcn) & 
                                   (roi_coords['task'] == 'loc') & 
                                   (roi_coords['roi'] == 'pIPS') &
                                   (roi_coords['hemisphere'] == 'left')].iloc[0]
            roi_ts = extract_roi_sphere(img4d, roi_coord[['x', 'y', 'z']].values.tolist())
            
            # Create searchlight object
            searchlight = SearchLight(
                brain_mask, 
                process_mask_img=brain_mask, 
                radius=6,  # You can adjust this radius as needed
                n_jobs=1, 
                verbose=1
            )
            
            # Define the function to be applied at each searchlight
            def searchlight_function(sphere_data, sphere_mask):
                sphere_ts = np.mean(sphere_data, axis=0).reshape(-1, 1)
                return gca_calculation(sphere_ts, roi_ts, psy)

            # Run searchlight
            searchlight_results = searchlight.fit(img4d, searchlight_function)

            # Save searchlight results
            searchlight_img = nib.Nifti1Image(searchlight_results.scores_, brain_mask.affine)
            nib.save(searchlight_img, f'{sub_dir}/derivatives/gca_searchlight/searchlight_results_{rc[0]}-{rc[-1]}.nii.gz')

            # Extract top N results for summary
            top_n = 100  # Adjust as needed
            flat_results = searchlight_results.scores_.ravel()
            top_indices = np.argsort(np.abs(flat_results))[-top_n:]
            for idx in top_indices:
                xyz = np.unravel_index(idx, searchlight_results.scores_.shape)
                f_diff = flat_results[idx]
                curr_data = pd.Series([ss, 'pIPS', xyz[0], xyz[1], xyz[2], 'loc', f_diff], index=sub_summary.columns)
                sub_summary = sub_summary.append(curr_data, ignore_index=True)

        logging.info(f'Completed GCA searchlight for subject {ss}')
        sub_summary.to_csv(f'{sub_dir}/derivatives/gca_searchlight/gca_searchlight_summary_{localizer.lower()}.csv', index=False)

    logging.info('GCA searchlight analysis completed for all subjects')

# Main execution
if __name__ == "__main__":
    conduct_gca_searchlight()