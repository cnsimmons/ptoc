# run ppi and fc

import os
import pandas as pd
import numpy as np
from nilearn import image, input_data
from nilearn.maskers import NiftiMasker
import nibabel as nib
from nilearn.glm.first_level import compute_regressor
import sys
import time

# Import your parameters
curr_dir = f'/user_data/csimmon2/git_repos/hemisphere'
sys.path.insert(0, curr_dir)

# Set up directories and parameters
study = 'hemispace'
study_dir = f"/lab_data/behrmannlab/vlad/{study}"
results_dir = '/user_data/csimmon2/git_repos/ptoc/results/tools/'

def extract_roi_timeseries(img, roi_mask, hemisphere='left'):
    """
    Extract mean time series from an anatomical ROI for a specific hemisphere
    """
    # Get the data
    roi_data = roi_mask.get_fdata()
    
    # Create hemisphere mask
    mid_x = roi_data.shape[0] // 2
    hemi_mask = np.zeros_like(roi_data)
    if hemisphere == 'left':
        hemi_mask[:mid_x, :, :] = 1
    else:  # right hemisphere
        hemi_mask[mid_x:, :, :] = 1
    
    # Combine ROI and hemisphere masks
    combined_mask = roi_data * hemi_mask
    mask_img = nib.Nifti1Image(combined_mask, roi_mask.affine)
    
    # Extract time series
    roi_masker = NiftiMasker(mask_img, standardize=True)
    seed_time_series = roi_masker.fit_transform(img)
    phys = np.mean(seed_time_series, axis=1).reshape(-1, 1)
    
    return phys, roi_masker

def make_psy_cov(runs, ss, raw_dir):
    """
    Create psychological regressor for PPI analysis using tool vs non-tool contrast
    """
    temp_dir = f'{raw_dir}/{ss}/ses-01'
    cov_dir = f'{temp_dir}/covs'
    vols, tr = 341, 1.0
    times = np.arange(0, vols * tr, tr)
    full_cov = pd.DataFrame(columns=['onset', 'duration', 'value'])

    for rn in runs:
        ss_num = ss.split('-')[1].replace('spaceloc', '')  # Extract just the number
        tool_cov_file = f'{cov_dir}/spaceloc_{ss_num}_run-0{rn}_tool.txt'
        nontool_cov_file = f'{cov_dir}/spaceloc_{ss_num}_run-0{rn}_non_tool.txt'

        if not os.path.exists(tool_cov_file) or not os.path.exists(nontool_cov_file):
            print(f'Covariate file not found for run {rn}')
            continue

        tool_cov = pd.read_csv(tool_cov_file, sep='\t', header=None, names=['onset', 'duration', 'value'])
        nontool_cov = pd.read_csv(nontool_cov_file, sep='\t', header=None, names=['onset', 'duration', 'value'])
        nontool_cov['value'] *= -1  # Reverse the sign for contrast
        full_cov = pd.concat([full_cov, tool_cov, nontool_cov])

    full_cov = full_cov.sort_values(by=['onset']).reset_index(drop=True)
    cov = full_cov.to_numpy()
    valid_onsets = cov[:, 0] < times[-1]
    cov = cov[valid_onsets]

    psy, _ = compute_regressor(cov.T, 'spm', times)
    return psy

def conduct_analyses(study_dir, raw_dir, sub, rois, run_num=2):
    """
    Main function to conduct FC and PPI analyses for a single subject
    """
    hemispheres = ['left', 'right']
    runs = list(range(1, run_num + 1))

    print(f"Processing subject: {sub}")
    sub_dir = f'{study_dir}/{sub}/ses-01/'
    roi_dir = f'{sub_dir}derivatives/rois'
    temp_dir = f'{raw_dir}/{sub}/ses-01/derivatives/fsl/toolloc'
    out_dir = f'{study_dir}/{sub}/ses-01/derivatives'
    os.makedirs(f'{out_dir}/fc', exist_ok=True)

    # Get subject-specific brain mask
    mask_path = f'{raw_dir}/{sub}/ses-01/anat/{sub}_ses-01_T1w_brain_mask.nii.gz'
    if not os.path.exists(mask_path):
        print(f"Brain mask not found for {sub}")
        return
        
    whole_brain_mask = nib.load(mask_path)
    brain_masker = NiftiMasker(whole_brain_mask, smoothing_fwhm=0, standardize=True)

    for tsk in ['loc']:
        for rr in rois:
            roi_path = f'{roi_dir}/parcels/{rr}.nii.gz'
            if not os.path.exists(roi_path):
                print(f"ROI file not found: {roi_path}")
                continue
            
            roi_img = nib.load(roi_path)
            
            for hemi in hemispheres:
                roi_start_time = time.time()
                print(f"Processing ROI: {rr}, Hemisphere: {hemi}")
                
                fc_file = f'{out_dir}/fc/{sub}_{rr}_{hemi}_{tsk}_fc_native.nii.gz'
                ppi_file = f'{out_dir}/fc/{sub}_{rr}_{hemi}_{tsk}_ppi_native.nii.gz'
                
                do_fc = not os.path.exists(fc_file)
                do_ppi = not os.path.exists(ppi_file)
                
                if not do_fc and not do_ppi:
                    print(f'Both FC and PPI files already exist. Skipping...')
                    continue

                # Load and concatenate both runs
                run_imgs = []
                for rn in runs:
                    run_path = f'{temp_dir}/run-0{rn}/1stLevel.feat/filtered_func_data_reg.nii.gz'
                    if os.path.exists(run_path):
                        img = image.clean_img(image.load_img(run_path), standardize=True)
                        run_imgs.append(img)
                
                if len(run_imgs) != 2:
                    print(f"Did not find both runs for {sub}")
                    continue
                    
                img4d = image.concat_imgs(run_imgs)
                
                # Extract ROI time series
                phys, _ = extract_roi_timeseries(img4d, roi_img, hemisphere=hemi)
                brain_time_series = brain_masker.fit_transform(img4d)
                
                if do_fc:
                    # FC Analysis
                    correlations = np.dot(brain_time_series.T, phys) / phys.shape[0]
                    correlation_img = brain_masker.inverse_transform(correlations)
                    nib.save(correlation_img, fc_file)
                    print(f'Saved FC result for {rr} {hemi}')
                
                if do_ppi:
                    # PPI Analysis
                    psy = make_psy_cov(runs, sub, raw_dir)
                    
                    # Ensure psy length matches phys
                    min_length = min(psy.shape[0], phys.shape[0], brain_time_series.shape[0])
                    psy = psy[:min_length]
                    phys = phys[:min_length]
                    brain_time_series = brain_time_series[:min_length]
                    
                    ppi_regressor = phys * psy
                    ppi_correlations = np.dot(brain_time_series.T, ppi_regressor) / ppi_regressor.shape[0]
                    ppi_img = brain_masker.inverse_transform(ppi_correlations)
                    nib.save(ppi_img, ppi_file)
                    print(f'Saved PPI result for {rr} {hemi}')
                
                print(f"Completed {rr} {hemi} in {time.time() - roi_start_time:.2f} seconds")

if __name__ == "__main__":
    study_dir = "/lab_data/behrmannlab/vlad/hemispace"
    raw_dir = "/lab_data/behrmannlab/vlad/hemispace"
    rois = ['LO', 'pIPS']
    
    if len(sys.argv) > 1:
        sub = sys.argv[1]
        conduct_analyses(study_dir, raw_dir, sub, rois, run_num=2)
    else:
        print("Please provide subject ID as command line argument")