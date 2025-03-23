# Schaefer Atlas 200 ROIs
# Create condition-specific connectivity matrices for each subject
import os
import sys
import pandas as pd
import numpy as np
import nibabel as nib
from nilearn import image, input_data, datasets
from nilearn.maskers import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
from nilearn.glm.first_level import compute_regressor
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import your parameters
curr_dir = f'/user_data/csimmon2/git_repos/ptoc'
sys.path.insert(0, curr_dir)
import ptoc_params as params

# Set up directories and parameters
study = 'ptoc'
study_dir = f"/lab_data/behrmannlab/vlad/{study}"
raw_dir = params.raw_dir
results_dir = '/user_data/csimmon2/git_repos/ptoc/results'

# Load subject information
sub_info = pd.read_csv(f'{curr_dir}/sub_info.csv')
subjects_to_skip = ['sub-084']
subs = sub_info[(sub_info['group'] == 'control') & (~sub_info['sub'].isin(subjects_to_skip))]['sub'].tolist()

run_num = 3
runs = list(range(1, run_num + 1))

# Define the number of ROIs for the Schaefer atlas
n_rois = 200

def verify_standard_space(img):
    """Verify image is in 2mm standard space"""
    if img.shape[:3] != (91, 109, 91):
        logging.warning(f"Unexpected shape: {img.shape}")
        return False
    
    vox_size = np.sqrt(np.sum(img.affine[:3, :3] ** 2, axis=0))
    if not np.allclose(vox_size, [2., 2., 2.], atol=0.1):
        logging.warning(f"Unexpected voxel size: {vox_size}")
        return False
    
    return True

def get_condition_mask(run_num, ss, condition, n_timepoints):
    """Create a binary mask for timepoints during a specific condition"""
    cov_dir = f'{raw_dir}/{ss}/ses-01/covs'
    ss_num = ss.split('-')[1]
    
    # Load condition timing file
    cov_file = f'{cov_dir}/catloc_{ss_num}_run-0{run_num}_{condition}.txt'
    if not os.path.exists(cov_file):
        logging.warning(f'Covariate file not found: {cov_file}')
        return np.zeros(n_timepoints, dtype=bool)
    
    # Load timing data
    cov = pd.read_csv(cov_file, sep='\t', header=None, 
                      names=['onset', 'duration', 'value'])
    
    # Create timepoints array
    tr = 2.0  # TR in seconds
    times = np.arange(0, n_timepoints * tr, tr)
    
    # Convert timing to binary mask
    condition_reg, _ = compute_regressor(cov.to_numpy().T, 'spm', times)
    
    # Convert to binary mask and ensure it's 1D
    return (condition_reg > 0).ravel()  # Added .ravel() to ensure 1D array

def create_condition_connectivity_matrix(ss, condition):
    """Create connectivity matrix for specific condition"""
    logging.info(f"Processing subject {ss} for condition {condition}")
    
    # Load Schaefer atlas
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=n_rois, yeo_networks=7, 
                                             resolution_mm=2)
    atlas_img = atlas.maps
    
    all_runs_data = []
    
    for rn in runs:
        # Load standard space data
        run_path = f'{raw_dir}/{ss}/ses-01/derivatives/reg_standard/filtered_func_run-0{rn}_standard.nii.gz'
        
        if not os.path.exists(run_path):
            logging.warning(f'Standard space data not found: {run_path}')
            continue
        
        subject_img = nib.load(run_path)
        
        # Verify standard space
        if not verify_standard_space(subject_img):
            logging.warning(f"Data not in expected standard space for {ss} run-{rn}")
            continue
        
        # Extract time series
        masker = NiftiLabelsMasker(
            labels_img=atlas_img,
            standardize='zscore_sample',
            memory=None,
            verbose=0
        )
        
        time_series = masker.fit_transform(subject_img)
        logging.info(f"Time series shape before masking: {time_series.shape}")
        
        # Get condition mask
        condition_mask = get_condition_mask(rn, ss, condition, time_series.shape[0])
        logging.info(f"Condition mask shape: {condition_mask.shape}")
        
        # Only keep timepoints during condition
        masked_time_series = time_series[condition_mask]
        logging.info(f"Time series shape after masking: {masked_time_series.shape}")
        
        if masked_time_series.shape[0] > 0:  # Only append if we have data
            all_runs_data.append(masked_time_series)
    
    if not all_runs_data:
        logging.warning(f'No valid data found for subject {ss} condition {condition}')
        return None
    
    # Concatenate runs
    full_time_series = np.concatenate(all_runs_data, axis=0)
    logging.info(f"Full time series shape: {full_time_series.shape}")
    
    # Compute connectivity matrix
    correlation_measure = ConnectivityMeasure(
        kind='correlation',
        standardize='zscore_sample'
    )
    connectivity_matrix = correlation_measure.fit_transform([full_time_series])[0]
    
    return connectivity_matrix

def main():
    conditions = ['Object', 'Scramble']
    
    for condition in conditions:
        # Create condition-specific output directory
        output_dir = f'{results_dir}/connectivity_matrices_{n_rois}_standard_{condition.lower()}'
        os.makedirs(output_dir, exist_ok=True)
        
        for ss in subs:
            try:
                connectivity_matrix = create_condition_connectivity_matrix(ss, condition)
                if connectivity_matrix is not None:
                    # Save matrix
                    output_path = f'{output_dir}/{ss}_connectivity_matrix_{condition.lower()}.npy'
                    np.save(output_path, connectivity_matrix)
                    logging.info(f'Saved {condition} connectivity matrix for {ss}')
                    
                    # Verify the saved matrix
                    loaded_matrix = np.load(output_path)
                    if not np.allclose(connectivity_matrix, loaded_matrix):
                        logging.warning(f'Matrix verification failed for {ss} {condition}')
                    
            except Exception as e:
                logging.error(f'Error processing subject {ss} condition {condition}: {str(e)}')
                continue

if __name__ == "__main__":
    main()