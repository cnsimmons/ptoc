import sys
import os
import logging
import warnings
import pandas as pd
import numpy as np
from nilearn import image, maskers
import nibabel as nib
from brainiak.searchlight.searchlight import Searchlight, Ball
import time
import gc
import resource

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up directories and parameters
study = 'ptoc'
study_dir = f"/lab_data/behrmannlab/vlad/{study}"
localizer = 'Object'  # or 'Scramble'
results_dir = '/user_data/csimmon2/git_repos/ptoc/results'
curr_dir = f'/user_data/csimmon2/git_repos/ptoc'
sys.path.insert(0, curr_dir)
import ptoc_params as params
raw_dir = params.raw_dir

# Load subject information
sub_info = pd.read_csv(f'{curr_dir}/sub_info.csv')
sub_info = sub_info[sub_info['group'] == 'control']
subs = ['sub-025']  # For testing, we're using just one subject
rois = ['pIPS', 'LO']
hemispheres = ['left', 'right']

run_num = 3
runs = list(range(1, run_num + 1))
run_combos = [[rn1, rn2] for rn1 in range(1, run_num + 1) for rn2 in range(rn1 + 1, run_num + 1)]

# Searchlight parameters
sl_rad = 2
max_blk_edge = 10
pool_size = 1
shape = Ball

def log_memory_usage(step):
    mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024  # Convert to MB
    logging.info(f"Memory usage after {step}: {mem_usage:.2f} MB")

def load_and_clean_run(run_path):
    curr_run = image.load_img(run_path)
    curr_run = image.clean_img(curr_run, standardize='zscore_sample')
    return curr_run

def extract_roi_sphere(img, coords):
    roi_masker = maskers.NiftiSpheresMasker([tuple(coords)], radius=6)
    seed_time_series = roi_masker.fit_transform(img)
    phys = np.mean(seed_time_series, axis=1).reshape(-1, 1)
    return phys

def make_psy_cov(runs, ss):
    # This is a placeholder function. In the future, it will create the psychological covariates.
    # For now, we'll just return a dummy array of the correct shape.
    vols_per_run = 184
    total_vols = vols_per_run * len(runs)
    return np.zeros((total_vols, 1))

def compute_gca(data, mask, myrad, bcast_var):
    # This is a placeholder function for GCA computation
    # Replace this with your actual GCA computation
    seed_ts, psy = bcast_var
    # Perform GCA computation here
    # For now, we'll just return a random value
    return np.array([np.random.rand()])

def conduct_searchlight():
    logging.info(f'Running searchlight analysis for {localizer}...')
    log_memory_usage("start")
    
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
            
            all_data = []
            all_masks = []
            for rn in rc:
                run_path = f'{exp_dir}/run-0{rn}/1stLevel.feat/filtered_func_data_reg.nii.gz'
                curr_run = load_and_clean_run(run_path)
                all_data.append(curr_run.get_fdata())
                
                brain_mask = image.load_img(f'{exp_dir}/run-0{rn}/1stLevel.feat/mask.nii.gz')
                all_masks.append(brain_mask.get_fdata() > 0)

            data = np.concatenate(all_data, axis=-1)
            mask = np.all(all_masks, axis=0)
            del all_data, all_masks
            gc.collect()
            log_memory_usage("data loaded and concatenated")

            # Extract seed time series (using pIPS as seed for this example)
            seed_coords = roi_coords[(roi_coords['roi'] == 'pIPS') & (roi_coords['hemisphere'] == 'left')]
            seed_ts = extract_roi_sphere(nib.Nifti1Image(data, curr_run.affine), seed_coords[['x', 'y', 'z']].values.tolist()[0])
            logging.info('Seed time series extracted')

            # Create psychological covariate (dummy for now)
            psy = make_psy_cov(rc, ss)
            logging.info('Psychological covariate created')

            # Create BrainIAK searchlight object
            sl = Searchlight(sl_rad=sl_rad, max_blk_edge=max_blk_edge, shape=shape)
            logging.info('Searchlight object created')
            log_memory_usage("before searchlight")

            # Run searchlight analysis
            logging.info("Starting searchlight analysis...")
            start_time = time.time()
            sl.distribute([data], mask)
            sl.broadcast((seed_ts, psy))
            sl_result = sl.run_searchlight(compute_gca, pool_size=pool_size)
            end_time = time.time()
            logging.info(f"Searchlight analysis completed in {end_time - start_time:.2f} seconds")
            log_memory_usage("after searchlight")

            # Save the searchlight result
            sl_result = sl_result.astype('double')
            sl_result[np.isnan(sl_result)] = 0
            result_img = nib.Nifti1Image(sl_result, curr_run.affine)
            nib.save(result_img, f'{sub_dir}/derivatives/gca/searchlight_result_{localizer.lower()}_rc{rcn}.nii.gz')

            logging.info(f"Completed searchlight for {ss}, run combination {rc}")

            # Clean up memory
            del data, mask, sl_result, seed_ts, psy
            gc.collect()
            log_memory_usage("end of iteration")

        logging.info(f'Completed searchlight analysis for subject {ss}')

    return "Searchlight analysis completed for all subjects"

if __name__ == "__main__":
    try:
        result = conduct_searchlight()
        logging.info(result)
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")