#!/usr/bin/env python3
"""
FC and PPI Analysis Script for Tool Localization

This script performs functional connectivity and psychophysiological interaction
analyses using ROI coordinates from a pre-processed file. It operates on subject 
data in native space and can process hemisphere-specific ROIs with flexible
contrast options.
"""

import sys
import os
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
raw_dir = '/user_data/csimmon2/git_repos/ptoc/raw_data'
results_dir = '/user_data/csimmon2/git_repos/ptoc/results'
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

def make_psy_cov(run, ss, condition='nontools', contrast_type='vs_scramble'):
    """
    Generate psychological covariates for PPI analysis
    
    Parameters:
    -----------
    run : int
        Run number
    ss : str
        Subject ID
    condition : str
        Condition to use as positive contrast (tools, nontools)
    contrast_type : str
        Type of contrast to use:
        - 'vs_scramble': Compare condition to scramble (default)
        - 'tools_vs_nontools': Compare tools to nontools
    
    Returns:
    --------
    psy : numpy.ndarray
        Psychological time series
    """
    logger = logging.getLogger(__name__)
    cov_dir = f'{raw_dir}/{ss}/ses-01/covs'
    times = np.arange(0, vols * tr, tr)
    subj_num = str(ss).replace("sub-spaceloc", "")
    
    # Define file paths based on contrast type
    if contrast_type == 'vs_scramble':
        # Standard condition vs scramble
        if condition == 'tools':
            pos_cov_file = f'{cov_dir}/ToolLoc_spaceloc{subj_num}_run{run}_tool.txt'
        elif condition == 'nontools':
            pos_cov_file = f'{cov_dir}/ToolLoc_spaceloc{subj_num}_run{run}_non_tool.txt'
        else:
            logger.error(f"Invalid condition: {condition}")
            return None
            
        neg_cov_file = f'{cov_dir}/ToolLoc_spaceloc{subj_num}_run{run}_scramble.txt'
        
    elif contrast_type == 'tools_vs_nontools':
        # Tools vs nontools
        pos_cov_file = f'{cov_dir}/ToolLoc_spaceloc{subj_num}_run{run}_tool.txt'
        neg_cov_file = f'{cov_dir}/ToolLoc_spaceloc{subj_num}_run{run}_non_tool.txt'
        
    else:
        logger.error(f"Invalid contrast type: {contrast_type}")
        return None
    
    # Check file existence
    if not os.path.exists(pos_cov_file):
        logger.error(f"Positive covariate file not found: {pos_cov_file}")
        return None
    if not os.path.exists(neg_cov_file):
        logger.error(f"Negative covariate file not found: {neg_cov_file}")
        return None
    
    # Load condition files
    pos_cov = pd.read_csv(pos_cov_file, sep='\t', header=None, names=['onset', 'duration', 'value'])
    neg_cov = pd.read_csv(neg_cov_file, sep='\t', header=None, names=['onset', 'duration', 'value'])
    
    # Negate the second condition
    neg_cov['value'] *= -1
    
    # Combine conditions
    full_cov = pd.concat([pos_cov, neg_cov])
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

def conduct_analysis(subjects, run_fc=False, run_ppi=False, condition='nontools', 
                     contrast_type='vs_scramble', force_overwrite=False):
    """
    Conduct FC and/or PPI analysis for specified subjects
    
    Parameters:
    -----------
    subjects : list
        List of subject IDs to process
    run_fc : bool
        If True, run functional connectivity analysis
    run_ppi : bool
        If True, run psychophysiological interaction analysis
    condition : str
        Condition to use (tools, nontools)
    contrast_type : str
        Type of contrast for PPI (vs_scramble, tools_vs_nontools)
    force_overwrite : bool
        If True, overwrites existing files; if False, skips subjects with existing files
    """
    logger = setup_logging()
    
    # Log analysis parameters
    logger.info(f"Starting analysis for {len(subjects)} subjects")
    logger.info(f"Analysis types - FC: {run_fc}, PPI: {run_ppi}")
    logger.info(f"Condition: {condition}, Contrast type: {contrast_type}")
    logger.info(f"Force overwrite: {force_overwrite}")
    
    # Check if roi_coordinates.csv exists
    if not os.path.exists(roi_coords_path):
        logger.error(f"ROI coordinates file not found: {roi_coords_path}")
        logger.error("Run extract_roi_coords.py first to generate this file")
        return
    
    # Define contrast name for file naming
    if contrast_type == 'vs_scramble':
        contrast_name = condition
    else:
        contrast_name = contrast_type
    
    # Load ROI coordinates
    roi_coords = pd.read_csv(roi_coords_path)
    
    # Process each subject
    for ss in subjects:
        logger.info(f"Processing subject: {ss}")
        
        # Define directories
        temp_dir = f'{raw_dir}/{ss}/ses-01/derivatives/fsl/toolloc'
        mask_path = f'{raw_dir}/{ss}/ses-01/anat/{ss}_ses-01_T1w_brain_mask.nii.gz'
        out_dir = f'/lab_data/behrmannlab/vlad/ptoc/{ss}/ses-01/derivatives'
        
        # Create output directories
        if run_fc:
            os.makedirs(f'{out_dir}/fc', exist_ok=True)
        if run_ppi:
            os.makedirs(f'{out_dir}/ppi', exist_ok=True)
        
        try:
            # Load brain mask
            whole_brain_mask = nib.load(mask_path)
            brain_masker = NiftiMasker(whole_brain_mask, standardize=True)
            
            # Process each ROI and hemisphere
            for roi in rois:
                for hemi in hemispheres:
                    hemi_prefix = hemi[0]
                    logger.info(f"Processing {roi} {hemi}")
                    
                    # Define file paths
                    fc_file = f'{out_dir}/fc/{ss}_{roi}_{hemi}_{contrast_name}_ToolLoc_fc.nii.gz'
                    ppi_file = f'{out_dir}/ppi/{ss}_{roi}_{hemi}_{contrast_name}_ToolLoc_ppi.nii.gz'
                    
                    # Skip if files exist and not forcing overwrite
                    if not force_overwrite:
                        if run_fc and os.path.exists(fc_file):
                            logger.info(f"Skipping FC for {ss} {roi} {hemi} - file exists (use --force to overwrite)")
                            run_fc_for_this_roi = False
                        else:
                            run_fc_for_this_roi = run_fc
                            
                        if run_ppi and os.path.exists(ppi_file):
                            logger.info(f"Skipping PPI for {ss} {roi} {hemi} - file exists (use --force to overwrite)")
                            run_ppi_for_this_roi = False
                        else:
                            run_ppi_for_this_roi = run_ppi
                            
                        if not run_fc_for_this_roi and not run_ppi_for_this_roi:
                            continue
                    else:
                        run_fc_for_this_roi = run_fc
                        run_ppi_for_this_roi = run_ppi
                    
                    all_runs_fc = []
                    all_runs_ppi = []
                    
                    # Process each run combination
                    for rcn, rc in enumerate(run_combos):
                        roi_run = rc[0]
                        analysis_run = rc[1]
                        
                        try:
                            # Get ROI coordinates for this subject, ROI, hemisphere, condition
                            # For PPI with tools vs nontools, we need coordinates matching either condition
                            # as specified in the roi_coordinates.csv file
                            coord_condition = condition
                            if contrast_type == 'tools_vs_nontools' and condition not in ['tools', 'nontools']:
                                # If a custom contrast name is used, default to using either tool or nontool coords
                                coord_condition = 'tools'  # or could use 'nontools'
                            
                            curr_coords = roi_coords[
                                (roi_coords['subject'] == ss) &
                                (roi_coords['run_combo'] == rcn) & 
                                (roi_coords['roi'] == f"{hemi_prefix}{roi}") &
                                (roi_coords['hemisphere'] == hemi_prefix) &
                                (roi_coords['condition'] == coord_condition)
                            ]
                            
                            if curr_coords.empty:
                                logger.warning(f"No coordinates found for {ss} {roi} {hemi} run_combo {rcn} condition {coord_condition}")
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
                            
                            # Get brain timeseries
                            brain_time_series = brain_masker.fit_transform(img)
                            
                            # FC Analysis
                            if run_fc_for_this_roi:
                                correlations = np.dot(brain_time_series.T, phys) / phys.shape[0]
                                correlations = np.arctanh(correlations.ravel())  # Fisher z-transform
                                correlation_img = brain_masker.inverse_transform(correlations)
                                all_runs_fc.append(correlation_img)
                                logger.info(f"Completed FC analysis for {ss} {roi} {hemi} run combo {rc[0]}->{rc[1]}")
                            
                            # PPI Analysis
                            if run_ppi_for_this_roi:
                                # Create psychological regressor
                                psy = make_psy_cov(analysis_run, ss, condition, contrast_type)
                                if psy is None:
                                    logger.error(f"Failed to create psychological regressor for {ss} run {analysis_run}")
                                    continue
                                
                                # Create PPI regressor (interaction term)
                                ppi = psy * phys
                                
                                # Compute correlations
                                seed_to_voxel_correlations = np.dot(brain_time_series.T, ppi) / ppi.shape[0]
                                seed_to_voxel_correlations = np.arctanh(seed_to_voxel_correlations.ravel())
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
                    
                    # Save the mean FC image across run combinations
                    if run_fc_for_this_roi and all_runs_fc:
                        logger.info(f"Saving FC file to {fc_file} with {len(all_runs_fc)} runs")
                        mean_fc = image.mean_img(all_runs_fc)
                        nib.save(mean_fc, fc_file)
                        logger.info(f"Saved FC image: {fc_file}")
                    elif run_fc_for_this_roi:
                        logger.warning(f"No FC data to save for {ss} {roi} {hemi}")
                    
                    # Save the mean PPI image across run combinations
                    if run_ppi_for_this_roi and all_runs_ppi:
                        logger.info(f"Saving PPI file to {ppi_file} with {len(all_runs_ppi)} runs")
                        mean_ppi = image.mean_img(all_runs_ppi)
                        nib.save(mean_ppi, ppi_file)
                        logger.info(f"Saved PPI image: {ppi_file}")
                    elif run_ppi_for_this_roi:
                        logger.warning(f"No PPI data to save for {ss} {roi} {hemi}")
        
        except Exception as e:
            logger.error(f"Error processing subject {ss}: {str(e)}")
            continue
    
    logger.info("Analysis completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run FC and/or PPI analysis for specific subjects')
    parser.add_argument('subjects', nargs='*', type=str, help='Subject IDs (e.g., sub-spaceloc1001 sub-spaceloc1002)')
    parser.add_argument('--all-subjects', action='store_true', help='Process all subjects from sub_info.csv')
    parser.add_argument('--fc', action='store_true', help='Run functional connectivity analysis')
    parser.add_argument('--ppi', action='store_true', help='Run psychophysiological interaction analysis')
    parser.add_argument('--condition', type=str, default='nontools', choices=['tools', 'nontools'], 
                        help='Condition to use (default: nontools)')
    parser.add_argument('--contrast', type=str, default='vs_scramble', choices=['vs_scramble', 'tools_vs_nontools'],
                        help='Contrast type for PPI (default: vs_scramble)')
    parser.add_argument('--force', action='store_true', help='Force overwrite existing files')
    
    args = parser.parse_args()
    
    # If neither --fc nor --ppi is specified, run both
    if not args.fc and not args.ppi:
        args.fc = True
        args.ppi = True
    
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
    
    conduct_analysis(subjects, 
                     run_fc=args.fc, 
                     run_ppi=args.ppi, 
                     condition=args.condition,
                     contrast_type=args.contrast,
                     force_overwrite=args.force)
    
    
'''

# Example usage:
# Run standard nontools vs scramble (both FC and PPI):
python fc_ppi.py sub-spaceloc1001

# Run only FC for tools vs scramble:
python fc_ppi.py sub-spaceloc1001 --fc --condition tools

# Run only PPI for tools vs nontools:
python fc_ppi.py sub-spaceloc1001 --ppi --contrast tools_vs_nontools

# Force overwrite existing files:
python fc_ppi.py sub-spaceloc1001 --force

# Run both FC and PPI on all subjects (nontools vs scramble):
python fc_ppi.py --all-subjects

# Run only PPI for tools vs nontools for all subjects:
python fc_ppi.py --all-subjects --ppi --contrast tools_vs_nontools

# Run only FC for tools vs scramble for all subjects:
python fc_ppi.py --all-subjects --fc --condition tools

# Run with forced overwrite:
python fc_ppi.py --all-subjects --force

'''