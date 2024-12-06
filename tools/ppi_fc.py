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

import os
import pandas as pd
from nilearn import image, plotting
import numpy as np
import nibabel as nib
import ptoc_params as params

# Set directories
raw_dir = params.raw_dir
sub_dir = '/user_data/csimmon2/temp_derivatives/{sub}/ses-01'
roi_dir = os.path.join(raw_dir, '{sub}/ses-01/derivatives/rois')
parcel_dir = os.path.join(raw_dir, '{sub}/ses-01/derivatives/rois/parcels')
output_dir = '/user_data/csimmon2/git_repos/ptoc/tools'

# Read subject information and filter for spaceloc
sub_info = pd.read_csv('/user_data/csimmon2/git_repos/ptoc/sub_info_tool.csv')
subs = sub_info[sub_info['exp'] == 'spaceloc']['sub'].tolist()

# Define parameters
parcels = ['pIPS', 'LO', 'PFS', 'aIPS']
run_combos = [[1, 2], [2, 1]]
zstats = {'tools': 3, 'scramble': 8}  # Dictionary to map condition names to zstat numbers

def extract_roi_coords():
    # Initialize DataFrame to store all results
    roi_coords = pd.DataFrame(columns=['subject', 'run_combo', 'task', 'condition', 'roi', 'hemisphere', 'x', 'y', 'z'])
    
    for ss in subs:
        os.makedirs(f"{sub_dir.format(sub=ss)}/spheres", exist_ok=True)

        for rcn, rc in enumerate(run_combos):
            for run_num in rc:
                # Process each condition (tools and scramble)
                for condition, zstat_num in zstats.items():
                    all_runs = [image.load_img(f"{sub_dir.format(sub=ss)}/derivatives/stats/zstat{zstat_num}_reg_run{run_num}.nii.gz")]
                    mean_zstat = image.mean_img(all_runs)
                    affine = mean_zstat.affine

                    for pr in parcels:
                        roi = image.load_img(f"{parcel_dir.format(sub=ss)}/{pr}.nii.gz")
                        roi_data = roi.get_fdata()
                        
                        # Create hemisphere masks
                        center_x = roi_data.shape[0] // 2
                        left_mask = np.zeros_like(roi_data)
                        right_mask = np.zeros_like(roi_data)
                        left_mask[:center_x, :, :] = 1
                        right_mask[center_x:, :, :] = 1
                        
                        for lr, hemi_mask in [('l', left_mask), ('r', right_mask)]:
                            hemi_roi_data = roi_data * hemi_mask
                            hemi_roi = image.new_img_like(roi, hemi_roi_data)
                            hemi_roi = image.math_img('img > 0', img=hemi_roi)
                            
                            if np.sum(hemi_roi.get_fdata()) == 0:
                                continue
                            
                            try:
                                # Create masked statistical map
                                masked_stat = image.math_img('img1 * img2', img1=hemi_roi, img2=mean_zstat)
                                masked_data = masked_stat.get_fdata()
                                
                                # Find peak coordinates using argmax
                                peak_idx = np.unravel_index(np.argmax(masked_data), masked_data.shape)
                                coords = image.coord_transform(peak_idx[0], peak_idx[1], peak_idx[2], mean_zstat.affine)
                                
                                # Add results to DataFrame
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
                                
                            except ValueError:
                                continue

    # Save all results to CSV
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'roi_coordinates.csv')
    roi_coords.to_csv(output_file, index=False)

def extract_roi_sphere(img, coords):
    roi_masker = input_data.NiftiSpheresMasker([tuple(coords)], radius=6)
    seed_time_series = roi_masker.fit_transform(img)
    return np.mean(seed_time_series, axis=1).reshape(-1, 1)

def make_psy_cov(run, ss, zstat_num):
    cov_dir = f'{raw_dir}/hemispace/{ss}/ses-01/covs'
    times = np.arange(0, vols * tr, tr)
    
    ss_num = ss.split('-')[1].replace('toolloc', '')
    tool_cov = pd.read_csv(f'{cov_dir}/ToolLoc_toolloc{ss_num}_run{run}_zstat{zstat_num}.txt', 
                          sep='\t', header=None, names=['onset', 'duration', 'value'])
    
    full_cov = tool_cov.sort_values(by=['onset'])
    cov = full_cov.to_numpy()
    psy, _ = compute_regressor(cov.T, 'spm', times)
    return psy

def conduct_analyses():
    logger = setup_logging()
    
    for ss in subs:
        logger.info(f"Processing subject: {ss}")
        
        temp_dir = f'{raw_dir}/hemispace/{ss}/ses-01/derivatives/fsl/toolloc'
        out_dir = f'/user_data/csimmon2/temp_derivatives/{ss}/ses-01/derivatives'
        os.makedirs(f'{out_dir}/fc', exist_ok=True)
        roi_coords = pd.read_csv(f'{out_dir}/spheres/sphere_coords_hemisphere_tools.csv')
        
        try:
            mask_path = f'{raw_dir}/hemispace/{ss}/ses-01/anat/{ss}_ses-01_T1w_brain_mask.nii.gz'
            whole_brain_mask = nib.load(mask_path)
            brain_masker = NiftiMasker(whole_brain_mask, standardize=True)
            
            for roi in rois:
                for hemi in hemispheres:
                    for contrast in ['object', 'scramble']:
                        logger.info(f"Processing {roi} {hemi} {contrast}")
                        
                        fc_file = f'{out_dir}/fc/{ss}_{roi}_{hemi}_ToolLoc_{contrast}_fc.nii.gz'
                        ppi_file = f'{out_dir}/fc/{ss}_{roi}_{hemi}_ToolLoc_{contrast}_ppi.nii.gz'
                        
                        all_runs_fc = []
                        all_runs_ppi = []
                        
                        for rcn, rc in enumerate(run_combos):
                            roi_run = rc[0]
                            analysis_run = rc[1]
                            
                            try:
                                curr_coords = roi_coords[
                                    (roi_coords['index'] == rcn) & 
                                    (roi_coords['task'] == f'ToolLoc_{contrast}') & 
                                    (roi_coords['roi'] == roi) &
                                    (roi_coords['hemisphere'] == hemi)
                                ]
                                
                                if curr_coords.empty:
                                    continue
                                    
                                coords = curr_coords[['x', 'y', 'z']].values.tolist()[0]
                                
                                img = image.clean_img(
                                    image.load_img(f'{temp_dir}/run-0{analysis_run}/1stLevel.feat/filtered_func_data_reg.nii.gz'),
                                    standardize=True
                                )
                                
                                phys = extract_roi_sphere(img, coords)
                                brain_time_series = brain_masker.fit_transform(img)
                                
                                # FC Analysis
                                correlations = np.dot(brain_time_series.T, phys) / phys.shape[0]
                                correlations = np.arctanh(correlations.ravel())
                                correlation_img = brain_masker.inverse_transform(correlations)
                                all_runs_fc.append(correlation_img)
                                
                                # PPI Analysis
                                zstat_num = 3 if contrast == 'object' else 8
                                psy = make_psy_cov(analysis_run, ss, zstat_num)
                                
                                min_length = min(psy.shape[0], phys.shape[0], brain_time_series.shape[0])
                                psy = psy[:min_length]
                                phys = phys[:min_length]
                                brain_time_series = brain_time_series[:min_length]
                                
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

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    logger = setup_logging()
    #extract_roi_coords() # completed and saved to roi_coordinates.csv
    conduct_analyses()