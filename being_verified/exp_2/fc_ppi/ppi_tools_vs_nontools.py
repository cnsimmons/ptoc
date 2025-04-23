#!/usr/bin/env python3
"""
PPI Analysis Script for Tools versus Nontools Contrast

This script performs psychophysiological interaction analysis specifically for 
tools versus nontools contrast using ROI coordinates from a pre-processed file.
It implements proper regression of main effects to ensure valid PPI results.
"""

import sys
sys.path.insert(0, '/user_data/csimmon2/git_repos/ptoc')
import os
import ptoc_params as params
import pandas as pd
from nilearn import image, input_data, plotting
import numpy as np
import nibabel as nib
import logging
import argparse
from nilearn.input_data import NiftiMasker
from nilearn.glm.first_level import compute_regressor
import time

# Settings
raw_dir = params.raw_dir
results_dir = params.results_dir
roi_coords_path = '/user_data/csimmon2/git_repos/ptoc/tools/roi_coordinates.csv'
sub_info_path = '/user_data/csimmon2/git_repos/ptoc/sub_info_tool.csv'

# Run parameters
tr = 1
vols = 341
run_num = 2
runs = list(range(1, run_num + 1))
run_combos = [[1,2], [2,1]]

# ROI settings
rois = ['pIPS', 'LO']
hemispheres = ['left', 'right']

def setup_logging():
    """Configure logging for the script"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def extract_roi_sphere(img, coords):
    """Extract time series from a spherical ROI centered at coordinates"""
    roi_masker = input_data.NiftiSpheresMasker([tuple(coords)], radius=6)
    seed_time_series = roi_masker.fit_transform(img)
    return np.mean(seed_time_series, axis=1).reshape(-1, 1)

def make_tool_vs_nontool_cov(run, ss):
    """
    Generate psychological covariates specifically for tools versus nontools contrast
    
    Parameters:
    -----------
    run : int
        Run number
    ss : str
        Subject ID
        
    Returns:
    --------
    psy : numpy.ndarray
        Psychological time series for tools vs nontools
    """
    logger = logging.getLogger(__name__)
    cov_dir = f'{raw_dir}/{ss}/ses-01/covs'
    times = np.arange(0, vols * tr, tr)
    subj_num = str(ss).replace("sub-spaceloc", "")
    
    # Define file paths for tools and nontools
    tool_cov_file = f'{cov_dir}/ToolLoc_spaceloc{subj_num}_run{run}_tool.txt'
    nontool_cov_file = f'{cov_dir}/ToolLoc_spaceloc{subj_num}_run{run}_non_tool.txt'
    
    # Check file existence
    if not os.path.exists(tool_cov_file):
        logger.error(f"Tool covariate file not found: {tool_cov_file}")
        return None
    if not os.path.exists(nontool_cov_file):
        logger.error(f"Nontool covariate file not found: {nontool_cov_file}")
        return None
    
    # Load tools condition as positive
    tool_cov = pd.read_csv(tool_cov_file, sep='\t', header=None, names=['onset', 'duration', 'value'])
    
    # Load nontools condition as negative
    nontool_cov = pd.read_csv(nontool_cov_file, sep='\t', header=None, names=['onset', 'duration', 'value'])
    nontool_cov['value'] *= -1  # Negate to represent the contrast
    
    # Combine conditions
    full_cov = pd.concat([tool_cov, nontool_cov])
    full_cov = full_cov.sort_values(by=['onset'])
    
    # Create regressor
    cov = full_cov.to_numpy()
    psy, _ = compute_regressor(cov.T, 'spm', times)
    return psy

def get_all_subjects():
    """Get all subject IDs from the subject info file"""
    logger = logging.getLogger(__name__)
    try:
        if not os.path.exists(sub_info_path):
            logger.error(f"Subject info file not found: {sub_info_path}")
            return []
            
        sub_info = pd.read_csv(sub_info_path)
        subs = sub_info[sub_info['exp'] == 'spaceloc']['sub'].tolist()
        return subs
    except Exception as e:
        logger.error(f"Error reading subject info file: {e}")
        return []

def conduct_tools_vs_nontools_ppi(subjects, force_overwrite=False):
    """
    Conduct PPI analysis specifically for tools versus nontools contrast
    
    Parameters:
    -----------
    subjects : list
        List of subject IDs to process
    force_overwrite : bool
        If True, overwrites existing files; if False, skips subjects with existing files
    """
    logger = setup_logging()
    
    # Log analysis parameters
    logger.info(f"Starting tools vs nontools PPI analysis for {len(subjects)} subjects")
    logger.info(f"Force overwrite: {force_overwrite}")
    
    # Define contrast name for file naming
    contrast_name = "tools_vs_nontools"
    
    # Check if roi_coordinates.csv exists
    if not os.path.exists(roi_coords_path):
        logger.error(f"ROI coordinates file not found: {roi_coords_path}")
        logger.error("Run extract_roi_coords.py first to generate this file")
        return
    
    # Load ROI coordinates
    roi_coords = pd.read_csv(roi_coords_path)
    
    # Process each subject
    for ss in subjects:
        logger.info(f"Processing subject: {ss}")
        
        # Define directories
        temp_dir = f'{raw_dir}/{ss}/ses-01/derivatives/fsl/toolloc'
        mask_path = f'{raw_dir}/{ss}/ses-01/anat/{ss}_ses-01_T1w_brain_mask.nii.gz'
        out_dir = f'/lab_data/behrmannlab/vlad/ptoc/{ss}/ses-01/derivatives'
        
        # Create output directory
        os.makedirs(f'{out_dir}/ppi', exist_ok=True)
        
        try:
            # Load brain mask
            whole_brain_mask = nib.load(mask_path)
            
            # Process each ROI and hemisphere
            for roi in rois:
                for hemi in hemispheres:
                    hemi_prefix = hemi[0]
                    logger.info(f"Processing {roi} {hemi}")
                    
                    # Define PPI output file
                    ppi_file = f'{out_dir}/ppi/{ss}_{roi}_{hemi}_{contrast_name}_ToolLoc_ppi.nii.gz'
                    
                    # Skip if file exists and not forcing overwrite
                    if not force_overwrite and os.path.exists(ppi_file):
                        logger.info(f"Skipping PPI for {ss} {roi} {hemi} - file exists (use --force to overwrite)")
                        continue
                    
                    all_runs_ppi = []
                    
                    # Process each run combination
                    for rcn, rc in enumerate(run_combos):
                        roi_run = rc[0]
                        analysis_run = rc[1]
                        
                        try:
                            # For tools vs nontools, use ROI coordinates from the tools condition
                            # You might need to adjust this based on your specific ROI definition strategy
                            curr_coords = roi_coords[
                                (roi_coords['subject'] == ss) &
                                (roi_coords['run_combo'] == rcn) & 
                                (roi_coords['roi'] == f"{hemi_prefix}{roi}") &
                                (roi_coords['hemisphere'] == hemi_prefix) &
                                (roi_coords['condition'] == 'tools')  # Using tools coordinates
                            ]
                            
                            if curr_coords.empty:
                                # Try nontools coordinates if tools not available
                                curr_coords = roi_coords[
                                    (roi_coords['subject'] == ss) &
                                    (roi_coords['run_combo'] == rcn) & 
                                    (roi_coords['roi'] == f"{hemi_prefix}{roi}") &
                                    (roi_coords['hemisphere'] == hemi_prefix) &
                                    (roi_coords['condition'] == 'nontools')  # Fall back to nontools coordinates
                                ]
                                
                                if curr_coords.empty:
                                    logger.warning(f"No coordinates found for {ss} {roi} {hemi} run_combo {rcn}")
                                    continue
                                
                            coords = [
                                curr_coords['x'].values[0],
                                curr_coords['y'].values[0],
                                curr_coords['z'].values[0]
                            ]
                            
                            # Load and preprocess functional image
                            func_img_path = f'{temp_dir}/run-0{analysis_run}/1stLevel.feat/filtered_func_data_reg.nii.gz'
                            if not os.path.exists(func_img_path):
                                logger.warning(f"Functional image not found: {func_img_path}")
                                continue
                                
                            img = image.clean_img(
                                image.load_img(func_img_path),
                                standardize=True
                            )
                            
                            # Extract ROI timeseries
                            phys = extract_roi_sphere(img, coords)
                            
                            # Create psychological regressor for tools vs nontools
                            psy = make_tool_vs_nontool_cov(analysis_run, ss)
                            if psy is None:
                                logger.error(f"Failed to create psychological regressor for {ss} run {analysis_run}")
                                continue
                            
                            # Create PPI regressor (interaction term)
                            ppi = psy * phys
                            
                            # Create confounds DataFrame with main effects
                            # This is critical for proper PPI analysis - regress out main effects
                            confounds = pd.DataFrame({
                                'psy': psy[:,0],
                                'phys': phys[:,0]
                            })
                            
                            # Create brain masker with proper confound regression
                            brain_masker = NiftiMasker(
                                whole_brain_mask, 
                                standardize=True,
                                detrend=True
                            )
                            
                            # Get brain timeseries with confound regression
                            # This is the key step that ensures proper PPI analysis
                            brain_time_series = brain_masker.fit_transform(img, confounds=confounds)
                            
                            # Compute correlations with PPI term
                            # This now represents the pure interaction effect
                            seed_to_voxel_correlations = np.dot(brain_time_series.T, ppi) / ppi.shape[0]
                            
                            # Fisher z-transform to normalize correlation values
                            seed_to_voxel_correlations = np.arctanh(seed_to_voxel_correlations.ravel())
                            
                            # Transform back to brain space
                            seed_to_voxel_correlations_img = brain_masker.inverse_transform(seed_to_voxel_correlations)
                            
                            # Save individual run PPI files
                            run_ppi_file = f'{out_dir}/ppi/{ss}_{roi}_{hemi}_{contrast_name}_ToolLoc_ppi_run{rc[0]}to{rc[1]}.nii.gz'
                            nib.save(seed_to_voxel_correlations_img, run_ppi_file)
                            logger.info(f"Saved individual run PPI file: {run_ppi_file}")
                            
                            all_runs_ppi.append(seed_to_voxel_correlations_img)
                            logger.info(f"Completed PPI analysis for {ss} {roi} {hemi} run combo {rc[0]}->{rc[1]}")
                            
                        except Exception as e:
                            logger.error(f"Error in run combo {rc}: {str(e)}")
                            continue
                    
                    # Save the mean PPI image across run combinations
                    if all_runs_ppi:
                        logger.info(f"Saving PPI file to {ppi_file} with {len(all_runs_ppi)} runs")
                        mean_ppi = image.mean_img(all_runs_ppi)
                        nib.save(mean_ppi, ppi_file)
                        logger.info(f"Saved PPI image: {ppi_file}")
                    else:
                        logger.warning(f"No PPI data to save for {ss} {roi} {hemi}")
        
        except Exception as e:
            logger.error(f"Error processing subject {ss}: {str(e)}")
            continue
    
    logger.info("Analysis completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Tools vs Nontools PPI analysis for specific subjects')
    parser.add_argument('subjects', nargs='*', type=str, help='Subject IDs (e.g., sub-spaceloc1001 sub-spaceloc1002)')
    parser.add_argument('--all-subjects', action='store_true', help='Process all subjects from sub_info.csv')
    parser.add_argument('--force', action='store_true', help='Force overwrite existing files')
    
    args = parser.parse_args()
    
    # Determine which subjects to process
    if args.all_subjects:
        subjects = get_all_subjects()
        if not subjects:
            print("Error: No subjects found or unable to read subject info file.")
            sys.exit(1)
    elif args.subjects:
        subjects = args.subjects
    else:
        print("Error: Either provide subject IDs or use --all-subjects flag.")
        parser.print_help()
        sys.exit(1)
    
    conduct_tools_vs_nontools_ppi(subjects, force_overwrite=args.force)
    

'''
# Run for specific subjects:
python tools_vs_nontools_ppi.py sub-spaceloc1001 sub-spaceloc1002

# Run for all subjects:
python tools_vs_nontools_ppi.py --all-subjects

# Force overwrite existing files:
python tools_vs_nontools_ppi.py --all-subjects --force
'''