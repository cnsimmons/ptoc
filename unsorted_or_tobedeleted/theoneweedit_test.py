##brainiak searchlight analysis for GCA
import sys
import os
import logging
import pandas as pd
import numpy as np
from nilearn import image, maskers
from nilearn.glm.first_level import compute_regressor
import nibabel as nib
from tqdm import tqdm
from brainiak.searchlight.searchlight import Searchlight, Ball
from mpi4py import MPI
import time
import gc

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

# Searchlight parameters
sl_rad = 2 #radius of searchlight sphere (in voxels)
max_blk_edge = 10 #how many blocks to send on each parallelized search
pool_size = 1 #number of cores to work on each search
voxels_proportion=1
shape = Ball

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

def compute_gca(data, seed_ts):
    # Placeholder function for GCA computation
    # Replace this with your actual GCA computation when ready
    return np.random.rand(1)

def conduct_searchlight():
    logging.info(f'Running searchlight analysis for {localizer}...')
    
    for ss in subs:
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
                curr_run = image.clean_img(curr_run, standardize='zscore_sample')
                filtered_list.append(curr_run)
                
                brain_mask = image.load_img(f'{exp_dir}/run-0{rn}/1stLevel.feat/mask.nii.gz')
                brain_masks.append(brain_mask)

            img4d = image.concat_imgs(filtered_list)
            logging.info(f"Concatenated image shape: {img4d.shape}")

            combined_mask_data = np.all([mask.get_fdata() > 0 for mask in brain_masks], axis=0)
            combined_brain_mask = nib.Nifti1Image(combined_mask_data.astype(np.int32), brain_masks[0].affine)
            logging.info('Brain mask has been created')

            # Extract seed time series (you'll need to implement this based on your ROI)
            seed_ts = extract_roi_sphere(img4d, roi_coords[['x', 'y', 'z']].values.tolist()[0])

            # Create BrainIAK searchlight object
            sl = Searchlight(sl_rad=sl_rad, max_blk_edge=max_blk_edge, shape=shape)
            
            # Prepare data for searchlight
            data = img4d.get_fdata()
            mask = combined_brain_mask.get_fdata().astype(bool)

            # Run searchlight analysis
            logging.info("Starting searchlight analysis...")
            start_time = time.time()
            sl.distribute([data], mask)
            sl.broadcast(seed_ts)
            sl_result = sl.run_searchlight(compute_gca, pool_size=pool_size)
            end_time = time.time()
            logging.info(f"Searchlight analysis completed in {end_time - start_time:.2f} seconds")

            # Save the searchlight result
            sl_result = sl_result.astype('double')
            sl_result[np.isnan(sl_result)] = 0
            result_img = nib.Nifti1Image(sl_result, img4d.affine)
            nib.save(result_img, f'{sub_dir}/derivatives/gca/searchlight_result_{localizer.lower()}_rc{rcn}.nii.gz')

            logging.info(f"Completed searchlight for {ss}, run combination {rc}")

            # Clean up memory
            del data, mask, sl_result
            gc.collect()

        logging.info(f'Completed searchlight analysis for subject {ss}')

    return "Searchlight analysis completed for all subjects"

if __name__ == "__main__":
    conduct_searchlight()