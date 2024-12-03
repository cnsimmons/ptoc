# PPI_FC for ToolLoc data
import sys
sys.path.insert(0, '/user_data/csimmon2/git_repos/ptoc')
import pandas as pd
import gc
from nilearn import image, plotting, input_data, glm
import numpy as np
from nilearn.input_data import NiftiMasker
import nibabel as nib
import os
from nilearn.glm import first_level
import warnings
from nilearn import image, input_data
from nilearn.glm.first_level import compute_regressor
import ptoc_params as params
import argparse

raw_dir = params.raw_dir
results_dir = params.results_dir
warnings.filterwarnings('ignore')

# Analysis parameters
run_num = 2
tr = 1
vols = 341

def extract_roi_timeseries(img, roi_mask, hemisphere='left'):
    """Extract ROI timeseries using parcels and hemisphere masking."""
    roi_data = roi_mask.get_fdata()
    mid_x = roi_data.shape[0] // 2
    
    hemi_mask = np.zeros_like(roi_data)
    if hemisphere == 'left':
        hemi_mask[:mid_x, :, :] = 1
    else:
        hemi_mask[mid_x:, :, :] = 1
    
    combined_mask = roi_data * hemi_mask
    mask_img = nib.Nifti1Image(combined_mask, roi_mask.affine)
    
    roi_masker = input_data.NiftiMasker(mask_img, standardize=True)
    seed_time_series = roi_masker.fit_transform(img)
    return np.mean(seed_time_series, axis=1).reshape(-1, 1)

def make_psy_cov(runs, ss, temp_dir):
    """Create psychological covariates for specified runs."""
    cov_dir = f'{temp_dir}/covs'
    times = np.arange(0, vols * len(runs), tr)
    full_cov = pd.DataFrame(columns=['onset', 'duration', 'value'])

    for run in runs:
        ss_num = ss.split('-')[1].replace('spaceloc', '')
        tool_cov_file = f'{cov_dir}/ToolLoc_spaceloc{ss_num}_run{run}_tool.txt'
        nontool_cov_file = f'{cov_dir}/ToolLoc_spaceloc{ss_num}_run{run}_non_tool.txt'

        if os.path.exists(tool_cov_file) and os.path.exists(nontool_cov_file):
            tool_cov = pd.read_csv(tool_cov_file, sep='\t', header=None, 
                                 names=['onset', 'duration', 'value'])
            nontool_cov = pd.read_csv(nontool_cov_file, sep='\t', header=None, 
                                    names=['onset', 'duration', 'value'])
            nontool_cov['value'] *= -1

            run_offset = vols * runs.index(run)
            tool_cov['onset'] += run_offset
            nontool_cov['onset'] += run_offset
            
            full_cov = pd.concat([full_cov, tool_cov, nontool_cov])

    full_cov = full_cov.sort_values(by=['onset'])
    cov = full_cov.to_numpy()
    psy, _ = compute_regressor(cov.T, 'spm', times)
    return psy

def conduct_analyses(sub, rois=['LO', 'pIPS']):
    """Main analysis function processing run combinations separately."""
    temp_dir = f'{raw_dir}/{sub}/ses-01/derivatives/fsl/toolloc'
    roi_dir = f'{raw_dir}/{sub}/ses-01/derivatives/rois'
    out_dir = f'/user_data/csimmon2/temp_derivatives/{sub}/ses-01/derivatives'
    os.makedirs(f'{out_dir}/fc', exist_ok=True)
    
    # Load brain mask
    mask_path = f'{raw_dir}/{sub}/ses-01/anat/{sub}_ses-01_T1w_brain_mask.nii.gz'
    whole_brain_mask = nib.load(mask_path)
    brain_masker = input_data.NiftiMasker(whole_brain_mask, standardize=True)

    run_combos = [[rn1, rn2] for rn1 in range(1, run_num + 1) 
                  for rn2 in range(rn1 + 1, run_num + 1)]

    for rr in rois:
        roi_path = f'{roi_dir}/parcels/{rr}.nii.gz'
        roi_img = nib.load(roi_path)
        
        for hemi in ['left', 'right']:
            print(f"Processing ROI: {rr}, Hemisphere: {hemi}")
            
            fc_file = f'{out_dir}/fc/{sub}_{rr}_{hemi}_toolloc_fc_native.nii.gz'
            ppi_file = f'{out_dir}/fc/{sub}_{rr}_{hemi}_toolloc_ppi_native.nii.gz'
            
            do_fc = not os.path.exists(fc_file)
            do_ppi = not os.path.exists(ppi_file)
            
            if not do_fc and not do_ppi:
                print(f'Both FC and PPI files exist for {rr} {hemi}. Skipping...')
                continue
            
            all_runs_fc = []
            all_runs_ppi = []
            
            for rc in run_combos:
                print(f"Processing run combination: {rc}")
                
                filtered_list = []
                for run in rc:
                    curr_run = image.load_img(
                        f'{temp_dir}/run-0{run}/1stLevel.feat/filtered_func_data_reg.nii.gz')
                    curr_run = image.clean_img(curr_run, standardize=True)
                    filtered_list.append(curr_run)

                img4d = image.concat_imgs(filtered_list)
                phys = extract_roi_timeseries(img4d, roi_img, hemisphere=hemi)
                brain_time_series = brain_masker.fit_transform(img4d)
                
                if do_fc:
                    correlations = np.dot(brain_time_series.T, phys) / phys.shape[0]
                    correlations = np.arctanh(correlations)
                    correlations = correlations.reshape(1, -1)
                    correlation_img = brain_masker.inverse_transform(correlations)
                    all_runs_fc.append(correlation_img)
                
                if do_ppi:
                    psy = make_psy_cov(rc, sub, f'{raw_dir}/{sub}/ses-01')
                    
                    min_length = min(psy.shape[0], phys.shape[0])
                    psy = psy[:min_length]
                    phys = phys[:min_length]
                    brain_time_series = brain_time_series[:min_length]
                    
                    ppi = psy * phys
                    correlations = np.dot(brain_time_series.T, ppi) / ppi.shape[0]
                    correlations = np.arctanh(correlations)
                    correlations = correlations.reshape(1, -1)
                    correlation_img = brain_masker.inverse_transform(correlations)
                    all_runs_ppi.append(correlation_img)
                
                gc.collect()
            
            if do_fc and all_runs_fc:
                mean_fc = image.mean_img(all_runs_fc)
                nib.save(mean_fc, fc_file)
                print(f'Saved FC result for {rr} {hemi}')
            
            if do_ppi and all_runs_ppi:
                mean_ppi = image.mean_img(all_runs_ppi)
                nib.save(mean_ppi, ppi_file)
                print(f'Saved PPI result for {rr} {hemi}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run PPI and FC analyses for a subject')
    parser.add_argument('subject', type=str, help='Subject ID (e.g., sub-spaceloc1002)')
    args = parser.parse_args()
    
    conduct_analyses(args.subject)