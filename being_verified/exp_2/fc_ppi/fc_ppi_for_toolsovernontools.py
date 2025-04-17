#fc_ppi native space with hemispheres

import sys
sys.path.insert(0, '/user_data/csimmon2/git_repos/ptoc')
import glob
import pandas as pd
import gc
from nilearn import image, input_data, plotting
import numpy as np
import nibabel as nib
import os
from nilearn.glm.first_level import compute_regressor
import warnings
import ptoc_params as params
import time
from nilearn.input_data import NiftiMasker
import logging

# Settings
raw_dir = params.raw_dir
results_dir = params.results_dir
sub_info_path = '/user_data/csimmon2/git_repos/ptoc/sub_info_tool.csv'

# Load subject info
sub_info = pd.read_csv(sub_info_path)
subs = sub_info[sub_info['exp'] == 'spaceloc']['sub'].tolist()

rois = ['pIPS', 'LO']
hemispheres = ['left', 'right']
condition = 'toolovernontool' # ['scramble', 'nontools', 'nontoolovertool', 'tools', 'toolovernontool']

# Run parameters
tr = 1
vols = 341
run_num = 2
runs = list(range(1, run_num + 1))
run_combos = [[1,2], [2,1]]

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

# Set directories
sub_dir = '/user_data/csimmon2/temp_derivatives/'
parcel_dir = os.path.join(raw_dir, '{sub}/ses-01/derivatives/rois/parcels')
output_dir = '/user_data/csimmon2/git_repos/ptoc/tools'

# Define parameters
zstats = {'tools': 3, 'scramble': 8, 'nontools': 4, 'toolovernontool': 1, 'nontoolovertool': 2}  # Dictionary to map condition names to zstat numbers


def extract_roi_sphere(img, coords):
    roi_masker = input_data.NiftiSpheresMasker([tuple(coords)], radius=6)
    seed_time_series = roi_masker.fit_transform(img)
    return np.mean(seed_time_series, axis=1).reshape(-1, 1)

def make_psy_cov(run, ss):
    """Generate psychological covariates combining tool and nontool conditions"""
    cov_dir = f'{raw_dir}/{ss}/ses-01/covs'
    times = np.arange(0, vols * tr, tr)
    
    # Load condition ## be sure to change.
    tool_cov = pd.read_csv(f'{cov_dir}/ToolLoc_spaceloc{str(ss).replace("sub-spaceloc","")}_run{run}_tool.txt', 
                          sep='\t', header=None, names=['onset', 'duration', 'value'])
    
    # Load and negate nontool condition
    nontool_cov = pd.read_csv(f'{cov_dir}/ToolLoc_spaceloc{str(ss).replace("sub-spaceloc","")}_run{run}_non_tool.txt', 
                              sep='\t', header=None, names=['onset', 'duration', 'value'])
    nontool_cov['value'] *= -1
    
    # Combine conditions
    full_cov = pd.concat([tool_cov, nontool_cov])
    full_cov = full_cov.sort_values(by=['onset'])
    
    # Create regressor
    cov = full_cov.to_numpy()
    psy, _ = compute_regressor(cov.T, 'spm', times)
    return psy

def conduct_analyses(run_fc=True, run_ppi=True): # exp 1 version
    """Conduct FC and PPI analyses for all subjects and ROIs"""
    logger = setup_logging()
    
    for ss in subs:
        logger.info(f"Processing subject: {ss}")
        
        temp_dir = f'{raw_dir}/{ss}/ses-01/derivatives/fsl/toolloc'
        mask_path = f'{raw_dir}/{ss}/ses-01/anat/{ss}_ses-01_T1w_brain_mask.nii.gz'
        out_dir = f'/lab_data/behrmannlab/vlad/ptoc/{ss}/ses-01/derivatives'
        
        if run_fc:
            os.makedirs(f'{out_dir}/fc', exist_ok=True)
        if run_ppi:
            os.makedirs(f'{out_dir}/ppi', exist_ok=True)
        
        roi_coords = pd.read_csv(f'{output_dir}/roi_coordinates.csv')
        
        try:
            whole_brain_mask = nib.load(mask_path)
            brain_masker = NiftiMasker(whole_brain_mask, standardize=True)
            
            for roi in rois:
                for hemi in hemispheres:
                    hemi_prefix = hemi[0]
                    logger.info(f"Processing {roi} {hemi}")
                    
                    # File paths
                    fc_file = f'{out_dir}/fc/{ss}_{roi}_{hemi}_{condition}_ToolLoc_fc.nii.gz'
                    ppi_file = f'{out_dir}/ppi/{ss}_{roi}_{hemi}_{condition}_ToolLoc_ppi.nii.gz'
                    
                    # Split the checks
                    run_fc = not os.path.exists(fc_file)
                
                    
                    all_runs_fc = []
                    all_runs_ppi = []
                    
                    for rcn, rc in enumerate(run_combos):
                        roi_run = rc[0]
                        analysis_run = rc[1]
                        
                        try:
                            
                            curr_coords = roi_coords[
                                (roi_coords['subject'] == ss) &
                                (roi_coords['run_combo'] == rcn) & 
                                (roi_coords['roi'] == f"{hemi_prefix}{roi}") &
                                (roi_coords['hemisphere'] == hemi_prefix) &
                                (roi_coords['condition'] == condition)  # Make the selection explicit
                            ]
                            
                            if curr_coords.empty:
                                logger.warning(f"No coordinates found for {ss} {roi} {hemi} run_combo {rcn}")
                                continue
                                
                            coords = [
                                curr_coords['x'].values[0],
                                curr_coords['y'].values[0],
                                curr_coords['z'].values[0]
                            ]
                            
                            img = image.clean_img(
                                image.load_img(f'{temp_dir}/run-0{analysis_run}/1stLevel.feat/filtered_func_data_reg.nii.gz'),
                                standardize=True
                            )
                            
                            # Extract ROI timeseries
                            phys = extract_roi_sphere(img, coords)
                            
                            # Get brain timeseries with standardization (like exp 1)
                            brain_time_series = brain_masker.fit_transform(img)
                            
                            # FC Analysis
                            if run_fc:
                                correlations = np.dot(brain_time_series.T, phys) / phys.shape[0]
                                correlations = np.arctanh(correlations.ravel())
                                correlation_img = brain_masker.inverse_transform(correlations)
                                all_runs_fc.append(correlation_img)
                  
                            # PPI Analysis
                            if run_ppi:
                                psy = make_psy_cov(analysis_run, ss)
                                # Create PPI regressor first (interaction term)
                                ppi = psy * phys
                            
                                # Compute correlations
                                seed_to_voxel_correlations = np.dot(brain_time_series.T, ppi) / ppi.shape[0]
                                
                                # Fisher z-transform
                                seed_to_voxel_correlations = np.arctanh(seed_to_voxel_correlations.ravel())
                                
                                # Transform back to brain space
                                seed_to_voxel_correlations_img = brain_masker.inverse_transform(seed_to_voxel_correlations)
                                
                                # Save individual run PPI files
                                run_ppi_file = f'{out_dir}/ppi/{ss}_{roi}_{hemi}_{condition}_ToolLoc_ppi_run{rc[0]}to{rc[1]}.nii.gz'
                                nib.save(seed_to_voxel_correlations_img, run_ppi_file)

                                all_runs_ppi.append(seed_to_voxel_correlations_img)
                            
                        except Exception as e:
                            logger.error(f"Error in run combo {rc}: {str(e)}")
                            continue
                    
                    if all_runs_fc:
                        mean_fc = image.mean_img(all_runs_fc)
                        nib.save(mean_fc, fc_file)
                    
                    if all_runs_ppi:
                        mean_ppi = image.mean_img(all_runs_ppi)
                        nib.save(mean_ppi, ppi_file)
        
        except Exception as e:
            logger.error(f"Error processing subject {ss}: {str(e)}")
            continue

# # to run all subs   
if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    logger = setup_logging()
    conduct_analyses(run_fc=False, run_ppi=True)
    #create_summary()