# The point of this script is to conduct a partial correlation analysis
# between pIPS and the rest of the brain, while regressing out LO activity.
# do not forget to run native2mni.py after this script to transform results to MNI space

import os
import sys
import pandas as pd
import numpy as np
import nibabel as nib
from nilearn import image
from nilearn.input_data import NiftiMasker
import time

# Import parameters
curr_dir = f'/user_data/csimmon2/git_repos/ptoc'
sys.path.insert(0, curr_dir)
import ptoc_params as params

# Set up directories and parameters
study = 'ptoc'
study_dir = f"/lab_data/behrmannlab/vlad/{study}"
raw_dir = params.raw_dir

# Load subject information
sub_info = pd.read_csv(f'{curr_dir}/sub_info.csv')
subs = sub_info[sub_info['group'] == 'control']['sub'].tolist()

hemispheres = ['left', 'right']
run_combos = [[1, 2], [1, 3], [2, 3]]

def extract_roi_sphere(img, coords, radius=6):
    """Extract timeseries from spherical ROI"""
    from nilearn.input_data import NiftiSpheresMasker
    sphere_masker = NiftiSpheresMasker([coords], radius=radius, standardize=True)
    time_series = sphere_masker.fit_transform(img)
    return time_series.flatten()

def conduct_analyses():
    for ss in subs:
        print(f"Processing subject: {ss}")
        sub_dir = f'{study_dir}/{ss}/ses-01/'
        roi_dir = f'{sub_dir}derivatives/rois'
        temp_dir = f'{raw_dir}/{ss}/ses-01/derivatives/fsl/loc'
        
        # Load ROI coordinates
        roi_coords = pd.read_csv(f'{roi_dir}/spheres/sphere_coords_hemisphere.csv')
        
        out_dir = f'{study_dir}/{ss}/ses-01/derivatives'
        os.makedirs(f'{out_dir}/fc', exist_ok=True)
        
        # subject-specific brain mask
        def get_subject_mask(ss):
            mask_path = f'{raw_dir}/{ss}/ses-01/anat/{ss}_ses-01_T1w_brain_mask.nii.gz'
            return nib.load(mask_path)
        
        # Load subject-specific mask
        whole_brain_mask = get_subject_mask(ss)
        brain_masker = NiftiMasker(whole_brain_mask, smoothing_fwhm=0, standardize=True)

        for tsk in ['loc']:
            for hemi in hemispheres:
                roi_start_time = time.time()
                print(f"Processing Hemisphere: {hemi}")
                
                # Output file
                fc_file = f'{out_dir}/fc/{ss}_pIPS_clean_{hemi}_{tsk}_fc.nii.gz'
                
                if os.path.exists(fc_file):
                    print(f'FC file for {hemi} already exists. Skipping...')
                    continue
                
                all_runs_fc = []
                
                for rcn, rc in enumerate(run_combos):
                    combo_start_time = time.time()
                    print(f"Processing run combination {rc} for {hemi}")
                    
                    # Get coordinates for both pIPS and LO
                    pips_coords = roi_coords[(roi_coords['index'] == rcn) & 
                                             (roi_coords['task'] == tsk) & 
                                             (roi_coords['roi'] == 'pIPS') &
                                             (roi_coords['hemisphere'] == hemi)]
                    
                    lo_coords = roi_coords[(roi_coords['index'] == rcn) & 
                                           (roi_coords['task'] == tsk) & 
                                           (roi_coords['roi'] == 'LO') &
                                           (roi_coords['hemisphere'] == hemi)]
                    
                    if pips_coords.empty or lo_coords.empty:
                        print(f"Missing coordinates for pIPS or LO, {hemi}, run combo {rc}")
                        continue
                    
                    pips_xyz = pips_coords[['x', 'y', 'z']].values.tolist()[0]
                    lo_xyz = lo_coords[['x', 'y', 'z']].values.tolist()[0]
                    
                    # Load and concatenate functional data
                    filtered_list = [image.clean_img(image.load_img(f'{temp_dir}/run-0{rn}/1stLevel.feat/filtered_func_data_reg.nii.gz'), standardize=True) for rn in rc]
                    img4d = image.concat_imgs(filtered_list)
                    
                    # Extract timeseries from both ROIs
                    dorsal_phys = extract_roi_sphere(img4d, pips_xyz)
                    ventral_phys = extract_roi_sphere(img4d, lo_xyz)
                    
                    # Ensure both signals have same length
                    min_length = min(len(dorsal_phys), len(ventral_phys))
                    dorsal_phys = dorsal_phys[:min_length]
                    ventral_phys = ventral_phys[:min_length]
                    
                    # Residualize ventral out of dorsal
                    beta = np.dot(ventral_phys.T, dorsal_phys) / np.dot(ventral_phys.T, ventral_phys)
                    dorsal_clean = dorsal_phys - beta * ventral_phys
                    
                    # Get whole brain timeseries
                    brain_time_series = brain_masker.fit_transform(img4d)
                    
                    # Ensure brain timeseries matches cleaned dorsal length
                    brain_time_series = brain_time_series[:min_length]
                    
                    # FC Analysis with cleaned dorsal signal
                    correlations = np.dot(brain_time_series.T, dorsal_clean) / dorsal_clean.shape[0]
                    correlations = np.arctanh(correlations.ravel())
                    correlation_img = brain_masker.inverse_transform(correlations)
                    all_runs_fc.append(correlation_img)
                    
                    combo_end_time = time.time()
                    print(f"Completed run combination {rc} for {hemi} in {combo_end_time - combo_start_time:.2f} seconds")
                
                # Average across run combinations
                if all_runs_fc:
                    mean_fc = image.mean_img(all_runs_fc)
                    nib.save(mean_fc, fc_file)
                    print(f'Saved FC result for {hemi}')
                else:
                    print(f"No valid run combinations for {hemi}")
                        
                roi_end_time = time.time()
                print(f"Completed {hemi} in {roi_end_time - roi_start_time:.2f} seconds")

# Call the function
if __name__ == "__main__":
    conduct_analyses()