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
import argparse

# Settings
raw_dir = params.raw_dir
results_dir = params.results_dir
sub_info_path = '/user_data/csimmon2/git_repos/ptoc/sub_info_tool.csv'

# Load subject info
sub_info = pd.read_csv(sub_info_path)
rois = ['pIPS', 'LO', 'PFS', 'aIPS']
hemispheres = ['left', 'right']

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
sub_dir = '/user_data/csimmon2/temp_derivatives/{sub}/ses-01'
roi_dir = os.path.join(raw_dir, '{sub}/ses-01/derivatives/rois')
parcel_dir = os.path.join(raw_dir, '{sub}/ses-01/derivatives/rois/parcels')
output_dir = '/user_data/csimmon2/git_repos/ptoc/tools'

# Define parameters
parcels = ['pIPS', 'LO', 'PFS', 'aIPS']
zstats = {'tools': 3, 'scramble': 8}


def extract_roi_sphere(img, coords):
    roi_masker = input_data.NiftiSpheresMasker([tuple(coords)], radius=6)
    seed_time_series = roi_masker.fit_transform(img)
    return np.mean(seed_time_series, axis=1).reshape(-1, 1)

def make_psy_cov(run, ss):
    cov_dir = f'{raw_dir}/{ss}/ses-01/covs'
    times = np.arange(0, vols * tr, tr)
    
    tool_cov = pd.read_csv(f'{cov_dir}/ToolLoc_spaceloc{str(ss).replace("sub-spaceloc","")}_run{run}_tool.txt', 
                          sep='\t', header=None, names=['onset', 'duration', 'value'])
    
    scramble_cov = pd.read_csv(f'{cov_dir}/ToolLoc_spaceloc{str(ss).replace("sub-spaceloc","")}_run{run}_scramble.txt', 
                              sep='\t', header=None, names=['onset', 'duration', 'value'])
    scramble_cov['value'] *= -1
    
    full_cov = pd.concat([tool_cov, scramble_cov])
    full_cov = full_cov.sort_values(by=['onset'])
    
    cov = full_cov.to_numpy()
    psy, _ = compute_regressor(cov.T, 'spm', times)
    return psy

def conduct_analyses(subjects=None):
    logger = setup_logging()
    subjects = subjects or subs
    
    for ss in subjects:
        logger.info(f"Processing subject: {ss}")
        
        temp_dir = f'{raw_dir}/{ss}/ses-01/derivatives/fsl/toolloc'
        mask_path = f'{raw_dir}/{ss}/ses-01/anat/{ss}_ses-01_T1w_brain_mask.nii.gz'
        out_dir = f'/user_data/csimmon2/temp_derivatives/{ss}/ses-01/derivatives'
        os.makedirs(f'{out_dir}/fc', exist_ok=True)
        
        roi_coords = pd.read_csv(f'{output_dir}/roi_coordinates.csv')
        
        try:
            whole_brain_mask = nib.load(mask_path)
            brain_masker = NiftiMasker(whole_brain_mask, standardize=True)
            
            for roi in rois:
                for hemi in hemispheres:
                    hemi_prefix = hemi[0]
                    logger.info(f"Processing {roi} {hemi}")
                    
                    fc_file = f'{out_dir}/fc/{ss}_{roi}_{hemi}_ToolLoc_fc.nii.gz'
                    ppi_file = f'{out_dir}/fc/{ss}_{roi}_{hemi}_ToolLoc_ppi.nii.gz'
                    
                    if os.path.exists(fc_file) and os.path.exists(ppi_file):
                        logger.info(f"Skipping {ss} {roi} {hemi} - already processed")
                        continue
                    
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
                            
                            phys = extract_roi_sphere(img, coords)
                            brain_time_series = brain_masker.fit_transform(img)
                            
                            correlations = np.dot(brain_time_series.T, phys) / phys.shape[0]
                            correlations = np.arctanh(correlations.ravel())
                            correlation_img = brain_masker.inverse_transform(correlations)
                            all_runs_fc.append(correlation_img)
                            
                            psy = make_psy_cov(analysis_run, ss)
                            
                            confounds = pd.DataFrame({
                                'psy': psy[:,0],
                                'phys': phys[:,0]
                            })
                            
                            brain_time_series = brain_masker.fit_transform(img, confounds=[confounds])
                            
                            ppi_regressor = phys * psy
                            ppi_correlations = np.dot(brain_time_series.T, ppi_regressor) / ppi_regressor.shape[0]
                            ppi_correlations = np.arctanh(ppi_correlations.ravel())
                            ppi_img = brain_masker.inverse_transform(ppi_correlations)
                            all_runs_ppi.append(ppi_img)
                        
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

def create_summary():
    summary_df = pd.DataFrame(columns=['sub'] + [f"{h}{r}" for h in hemispheres for r in rois])
    
    for ss in subs:
        roi_means = [ss]
        
        for target_hemi in hemispheres:
            for target_roi in rois:
                roi = f"{target_hemi[0]}{target_roi}"
                
                try:
                    roi_mask = image.load_img(f'{raw_dir}/{ss}/ses-01/derivatives/rois/{roi}.nii.gz')
                    roi_masker = input_data.NiftiMasker(roi_mask)
                    
                    ppi_img = image.load_img(f'{out_dir}/sub-{ss}_{roi}_fc.nii.gz')
                    
                    acts = roi_masker.fit_transform(ppi_img)
                    roi_means.append(acts.mean())
                    
                except Exception as e:
                    roi_means.append(np.nan)
                    
        summary_df = summary_df.append(pd.Series(roi_means, index=summary_df.columns), ignore_index=True)
    
    summary_df.to_csv(f'{results_dir}/roi_ppi_summary.csv', index=False)

def process_subject(subject_id):
    warnings.filterwarnings('ignore')
    logger = setup_logging()
    logger.info(f"Processing subject: {subject_id}")
    conduct_analyses([subject_id])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('subject_id', help='Subject ID to process')
    args = parser.parse_args()
    process_subject(args.subject_id)