import sys
import os
import logging
import pandas as pd
import numpy as np
from nilearn import image, maskers
import nibabel as nib
from brainiak.searchlight.searchlight import Searchlight, Ball
import time
import gc
import warnings

# suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up directories and parameters
study = 'ptoc'
study_dir = f"/lab_data/behrmannlab/vlad/{study}"
localizer = 'Object'
results_dir = '/user_data/csimmon2/git_repos/ptoc/results'
curr_dir = f'/user_data/csimmon2/git_repos/ptoc'
sys.path.insert(0, curr_dir)
import ptoc_params as params
raw_dir = params.raw_dir

# Load subject information
sub_info = pd.read_csv(f'{curr_dir}/sub_info.csv')
sub_info = sub_info[sub_info['group'] == 'control']
subs = ['sub-025']
rois = ['pIPS']
hemispheres = ['left']

'''
run_num = 3
runs = list(range(1, run_num + 1))
run_combos = [[rn1, rn2] for rn1 in range(1, run_num + 1) for rn2 in range(rn1 + 1, run_num + 1)]
'''

run =1 # for now

# Searchlight parameters
sl_rad = 2
max_blk_edge = 10
pool_size = 1
shape = Ball

def extract_roi_sphere(img, coords):
    roi_masker = maskers.NiftiSpheresMasker([tuple(coords)], radius=6)
    seed_time_series = roi_masker.fit_transform(img)
    phys = np.mean(seed_time_series, axis=1).reshape(-1, 1)
    return phys

def compute_gca(data, mask, myrad, bcast_var):
    # Placeholder function for GCA computation
    # This function now accepts the correct number of arguments
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

        # Since we're only using run 1 for now, we'll modify this part
        logging.info(f"Processing run 1 for subject {ss}")
        
        curr_run = image.load_img(f'{exp_dir}/run-01/1stLevel.feat/filtered_func_data_reg.nii.gz')
        curr_run = image.clean_img(curr_run, standardize='zscore_sample')
        
        brain_mask = image.load_img(f'{exp_dir}/run-01/1stLevel.feat/mask.nii.gz')

        img4d = curr_run
        logging.info(f"Image shape: {img4d.shape}")

        combined_brain_mask = brain_mask
        logging.info('Brain mask has been created')

        # Extract seed time series
        seed_ts = extract_roi_sphere(img4d, roi_coords[['x', 'y', 'z']].values.tolist()[0])
        logging.info('seed_ts extracted')

        # Create BrainIAK searchlight object
        sl = Searchlight(sl_rad=sl_rad, max_blk_edge=max_blk_edge, shape=shape)
        logging.info('Searchlight object created')
        
        # Prepare data for searchlight
        data = img4d.get_fdata()
        mask = combined_brain_mask.get_fdata().astype(bool)
        logging.info('data and mask prepared')

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
        nib.save(result_img, f'{sub_dir}/derivatives/gca/searchlight_result_{localizer.lower()}_run01.nii.gz')

        logging.info(f"Completed searchlight for {ss}, run 1")

        # Clean up memory
        del data, mask, sl_result
        gc.collect()

    logging.info(f'Completed searchlight analysis for all subjects')

    return "Searchlight analysis completed for all subjects"

if __name__ == "__main__":
    conduct_searchlight()