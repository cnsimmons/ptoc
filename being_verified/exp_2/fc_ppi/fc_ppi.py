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
condition = 'nontools' # ['scramble', 'nontools']

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

def extract_roi_coords():
    raw_dir = params.raw_dir
    parcels = ['pIPS', 'aIPS', 'LO']
    run_combos = [[1, 2], [2, 1]]
    #zstats = {'tools': 3, 'scramble': 8, 'nontools': 4}
    
    sub_info = pd.read_csv('/user_data/csimmon2/git_repos/ptoc/sub_info_tool.csv')
    subs = sub_info[sub_info['exp'] == 'spaceloc']['sub'].tolist()
    roi_coords = pd.DataFrame(columns=['subject', 'run_combo', 'task', 'condition', 'roi', 'hemisphere', 'x', 'y', 'z'])
    print(f"Initial DataFrame size: {len(roi_coords)}")
    
    for ss in subs:
        for rcn, rc in enumerate(run_combos):
            for condition, zstat_num in zstats.items():
                run_num = rc[0]
                #zstat_path = f"{raw_dir}/{ss}/ses-01/derivatives/fsl/toolloc/run-0{run_num}/1stLevel.feat/stats/zstat{zstat_num}_reg.nii.gz"
                zstat_path = f"{sub_dir}/{ss}/ses-01/derivatives/stats/zstat{zstat_num}_reg_run{run_num}.nii.gz"
                print(f"Processing {ss} - {condition} - Run {run_num}")
                
                if not os.path.exists(zstat_path):
                    print(f"Missing zstat file: {zstat_path}")
                    continue
                    
                try:
                    mean_zstat = image.load_img(zstat_path)
                    
                    for pr in parcels:
                        roi_path = f"{raw_dir}/{ss}/ses-01/derivatives/rois/parcels/{pr}.nii.gz"
                        print(f"Trying to access: {roi_path}")
                        if not os.path.exists(roi_path):
                            print(f"File not found: {roi_path}")
                            continue
                            
                        roi = image.load_img(roi_path)
                        roi_data = roi.get_fdata()
                        
                        center_x = roi_data.shape[0] // 2
                        for lr, slice_idx in [('l', slice(None, center_x)), ('r', slice(center_x, None))]:
                            hemi_data = np.zeros_like(roi_data)
                            hemi_data[slice_idx] = roi_data[slice_idx]
                            
                            if np.sum(hemi_data) == 0:
                                continue
                                
                            hemi_roi = image.new_img_like(roi, hemi_data)
                            hemi_roi = image.math_img('img > 0', img=hemi_roi)
                            
                            coords = plotting.find_xyz_cut_coords(mean_zstat, mask_img=hemi_roi, activation_threshold=0.99)
                            
                            new_row = pd.DataFrame({
                                'subject': [ss],
                                'run_combo': [rcn],
                                'task': ['ToolLoc'],
                                'condition': [condition],
                                'roi': [f"{lr}{pr}"],
                                'hemisphere': [lr],
                                'x': [coords[0]],
                                'y': [coords[1]],
                                'z': [coords[2]]
                            })
                            roi_coords = pd.concat([roi_coords, new_row], ignore_index=True)
                            
                except Exception as e:
                    print(f"Error processing {ss} run {run_num} {condition}: {e}")
                    continue

    output_dir = '/user_data/csimmon2/git_repos/ptoc/tools'
    os.makedirs(output_dir, exist_ok=True)
    roi_coords.to_csv(os.path.join(output_dir, 'roi_coordinates.csv'), index=False)

def extract_roi_sphere(img, coords):
    roi_masker = input_data.NiftiSpheresMasker([tuple(coords)], radius=6)
    seed_time_series = roi_masker.fit_transform(img)
    return np.mean(seed_time_series, axis=1).reshape(-1, 1)

def make_psy_cov(run, ss):
    """Generate psychological covariates combining tool and scramble conditions"""
    cov_dir = f'{raw_dir}/{ss}/ses-01/covs'
    times = np.arange(0, vols * tr, tr)
    
    # Load condition ## be sure to change.
    tool_cov = pd.read_csv(f'{cov_dir}/ToolLoc_spaceloc{str(ss).replace("sub-spaceloc","")}_run{run}_non_tool.txt', 
                          sep='\t', header=None, names=['onset', 'duration', 'value']) # change between tool and non_tool for relevant condition
    
    # Load and negate scramble condition
    scramble_cov = pd.read_csv(f'{cov_dir}/ToolLoc_spaceloc{str(ss).replace("sub-spaceloc","")}_run{run}_scramble.txt', 
                              sep='\t', header=None, names=['onset', 'duration', 'value'])
    scramble_cov['value'] *= -1
    
    # Combine conditions
    full_cov = pd.concat([tool_cov, scramble_cov])
    full_cov = full_cov.sort_values(by=['onset'])
    
    # Create regressor
    cov = full_cov.to_numpy()
    psy, _ = compute_regressor(cov.T, 'spm', times)
    return psy

def conduct_analyses(run_fc=True, run_ppi=False): # exp 1 version
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
                    
                    # Remove ppi check so it always runs
                    #if os.path.exists(fc_file) and os.path.exists(ppi_file):
                        #logger.info(f"Skipping {ss} {roi} {hemi} - already processed")
                        #continue
                    
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

                                # # Create confounds DataFrame with main effects || this section is now unnecessary because we're using a different method via standardized brain timeseries using brain_masker
                                #confounds = pd.DataFrame(columns=['psy', 'phys'])
                                #confounds['psy'] = psy[:,0]
                                #confounds['phys'] = phys[:,0]
                            
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
#if __name__ == "__main__":
    #warnings.filterwarnings('ignore')
    #logger = setup_logging()
    #extract_roi_coords() # completed and saved to roi_coordinates.csv
    #conduct_analyses(run_fc=True, run_ppi=True)
    #create_summary()

if __name__ == "__main__":
    import argparse
    
    ## Set up argument parser
    parser = argparse.ArgumentParser(description='Run FC and PPI analysis for specific subjects')
    parser.add_argument('subjects', nargs='+', type=str, help='Subject IDs (e.g., sub-spaceloc1001 sub-spaceloc1002)')

    ## Parse arguments
    args = parser.parse_args()
    
    ## Override the subs list with just the input subject
    subs = args.subjects
    
    warnings.filterwarnings('ignore')
    logger = setup_logging()
    #extract_roi_coords()
    conduct_analyses()
    #create_summary() # use visualize_fc_ppi.py to run create_summary

'''
def conduct_analyses_retro(run_fc=False, run_ppi=True): # exp VA version
"""Conduct FC and PPI analyses using explicit confound regression (Vlad approach)"""
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
                
                fc_file = f'{out_dir}/fc/{ss}_{roi}_{hemi}_ToolLoc_fc_retro.nii.gz'
                ppi_file = f'{out_dir}/ppi/{ss}_{roi}_{hemi}_ToolLoc_ppi_retro.nii.gz'
                
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
                            (roi_coords['hemisphere'] == hemi_prefix)
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
                        brain_time_series = brain_masker.fit_transform(img)

                        if run_fc:
                            # FC Analysis remains the same
                            correlations = np.dot(brain_time_series.T, phys) / phys.shape[0]
                            correlations = np.arctanh(correlations.ravel())
                            correlation_img = brain_masker.inverse_transform(correlations)
                            all_runs_fc.append(correlation_img)
                
                        if run_ppi:
                            # create psychological covariate
                            psy = make_psy_cov(analysis_run, ss)
                            
                            confounds = pd.DataFrame({
                                'psy': psy[:,0],
                                'phys': phys[:,0]
                            })
                            
                            brain_time_series = brain_masker.fit_transform(img, confounds=confounds) # does the location of this line make a major difference?
                        
                            # Create PPI regressor
                            ppi = psy * phys
                            
                            # Compute correlations
                            seed_to_voxel_correlations = np.dot(brain_time_series.T, ppi) / ppi.shape[0]
                            seed_to_voxel_correlations = np.arctanh(seed_to_voxel_correlations.ravel())
                            
                            # Transform back to brain space
                            seed_to_voxel_correlations_img = brain_masker.inverse_transform(seed_to_voxel_correlations)
                            all_runs_ppi.append(seed_to_voxel_correlations_img)
                        
                    except Exception as e:
                        logger.error(f"Error in run combo {rc}: {str(e)}")
                        continue
                
                if run_fc and all_runs_fc:
                    mean_fc = image.mean_img(all_runs_fc)
                    nib.save(mean_fc, fc_file)
                
                if all_runs_ppi:
                    # First save the individual run results || split by run so that we can plot sphere to sphere connectivity
                    for rcn, rc in enumerate(run_combos):
                        nib.save(all_runs_ppi[rcn], f'{out_dir}/ppi/{ss}_{roi}_{hemi}_ToolLoc_ppi_run{rc[0]}to{rc[1]}.nii.gz')
                    
                    # Then save the mean
                    mean_ppi = image.mean_img(all_runs_ppi)
                    nib.save(mean_ppi, ppi_file)
    
    except Exception as e:
        logger.error(f"Error processing subject {ss}: {str(e)}")
        continue
        
        
def create_summary(run_fc=False, run_ppi=True): #### will need to update after running tools ppi_fc create_summary to account for {condition} in the name
    """Extract average FC and/or PPI values using sphere-to-sphere connections"""
    logger = setup_logging()
    
    # Read the coordinates file that contains sphere centers
    roi_coords = pd.read_csv(f'{output_dir}/roi_coordinates.csv')
    
    # Initialize DataFrames based on what we're running
    columns = ['subject', 'seed_roi', 'seed_hemi', 'target_roi', 'target_hemi', 'run_combo', 'value']
    ppi_df = pd.DataFrame(columns=columns) if run_ppi else None
    fc_df = pd.DataFrame(columns=columns) if run_fc else None
    
    for ss in subs:
        logger.info(f"Processing subject: {ss}")
        
        out_dir = f'/lab_data/behrmannlab/vlad/ptoc/{ss}/ses-01/derivatives'
        
        # For each seed ROI
        for seed_roi in rois:
            for seed_hemi in hemispheres:
                seed_prefix = seed_hemi[0]
                
                # For each target ROI
                for target_roi in rois:
                    for target_hemi in hemispheres:
                        target_prefix = target_hemi[0]
                        
                        # Skip self-connections
                        if seed_roi == target_roi and seed_hemi == target_hemi:
                            continue
                            
                        # Process each run combination
                        for rcn, rc in enumerate(run_combos):
                            try:
                                # Get seed coordinates
                                seed_coords = roi_coords[
                                    (roi_coords['subject'] == ss) &
                                    (roi_coords['run_combo'] == rcn) & 
                                    (roi_coords['roi'] == f"{seed_prefix}{seed_roi}") &
                                    (roi_coords['hemisphere'] == seed_prefix)
                                ]
                                
                                # Get target coordinates
                                target_coords = roi_coords[
                                    (roi_coords['subject'] == ss) &
                                    (roi_coords['run_combo'] == rcn) & 
                                    (roi_coords['roi'] == f"{target_prefix}{target_roi}") &
                                    (roi_coords['hemisphere'] == target_prefix)
                                ]
                                
                                if seed_coords.empty or target_coords.empty:
                                    logger.warning(f"Missing coordinates for {ss} {seed_roi}-{target_roi} run_combo {rcn}")
                                    continue
                                
                                # Create target sphere masker
                                target_coords_xyz = [
                                    target_coords['x'].values[0],
                                    target_coords['y'].values[0],
                                    target_coords['z'].values[0]
                                ]
                                target_masker = input_data.NiftiSpheresMasker(
                                    [tuple(target_coords_xyz)], 
                                    radius=6,
                                    standardize=True
                                )
                                
                                # Extract PPI values if requested
                                if run_ppi:
                                    ppi_file = f'{out_dir}/ppi/{ss}_{seed_roi}_{seed_hemi}_ToolLoc_ppi_run{rc[0]}to{rc[1]}.nii.gz'
                                    if os.path.exists(ppi_file):
                                        ppi_img = image.load_img(ppi_file)
                                        ppi_value = target_masker.fit_transform(ppi_img).mean()
                                        
                                        new_row = pd.DataFrame({
                                            'subject': [ss],
                                            'seed_roi': [seed_roi],
                                            'seed_hemi': [seed_hemi],
                                            'target_roi': [target_roi],
                                            'target_hemi': [target_hemi],
                                            'run_combo': [f"{rc[0]}to{rc[1]}"],
                                            'value': [ppi_value]
                                        })
                                        ppi_df = pd.concat([ppi_df, new_row], ignore_index=True)
                                    else:
                                        logger.warning(f"Missing PPI file: {ppi_file}")
                                
                                # Extract FC values if requested
                                if run_fc:
                                    fc_file = f'{out_dir}/fc/{ss}_{seed_roi}_{seed_hemi}_ToolLoc_fc.nii.gz'
                                    if os.path.exists(fc_file):
                                        fc_img = image.load_img(fc_file)
                                        fc_value = target_masker.fit_transform(fc_img).mean()
                                        
                                        new_row = pd.DataFrame({
                                            'subject': [ss],
                                            'seed_roi': [seed_roi],
                                            'seed_hemi': [seed_hemi],
                                            'target_roi': [target_roi],
                                            'target_hemi': [target_hemi],
                                            'run_combo': [f"{rc[0]}to{rc[1]}"],
                                            'value': [fc_value]
                                        })
                                        fc_df = pd.concat([fc_df, new_row], ignore_index=True)
                                
                            except Exception as e:
                                logger.error(f"Error processing {ss} {seed_roi}-{target_roi} run {rc}: {str(e)}")
                                continue
    
    # Save results
    if run_ppi:
        if len(ppi_df) > 0:
            ppi_df.to_csv(f'{results_dir}/roi_ppi_sphere_summary.csv', index=False)
            logger.info(f"Saved PPI results with {len(ppi_df)} connections")
        else:
            logger.error("No PPI data was collected!")
    
    if run_fc:
        if len(fc_df) > 0:
            fc_df.to_csv(f'{results_dir}/roi_fc_sphere_summary.csv', index=False)
            logger.info(f"Saved FC results with {len(fc_df)} connections")
        else:
            logger.error("No FC data was collected!")
    
    return ppi_df if run_ppi else None, fc_df if run_fc else None
'''