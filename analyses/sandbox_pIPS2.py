import os
import pandas as pd
import numpy as np
from nilearn import image, maskers, plotting
from nilearn.maskers import NiftiMasker
from nilearn.datasets import load_mni152_brain_mask
from nilearn import datasets
from nilearn.glm.first_level import compute_regressor
import nibabel as nib
import sys
import time
import itertools 
import warnings

# Import your parameters
curr_dir = f'/user_data/csimmon2/git_repos/ptoc'
sys.path.insert(0, curr_dir)
import ptoc_params as params

# Set up directories and parameters
study = 'ptoc'
study_dir = f"/lab_data/behrmannlab/vlad/{study}"
ptoc_dir = f"/lab_data/behrmannlab/vlad/{study}"
results_dir = f'/user_data/csimmon2/git_repos/ptoc/results'
hemispace_dir = f'/lab_data/behrmannlab/vlad/hemispace' 
raw_dir = params.raw_dir
mni_parcel_dir = f'{curr_dir}/roiParcels' 

subs = ['sub-038', 'sub-057', 'sub-059', 'sub-064', 'sub-067']
rois = ['pIPS']

## Warning traceback
def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file, 'write') else sys.stderr
    traceback = sys.exc_info()[2]
    log.write(warnings.formatwarning(message, category, filename, lineno, line))
    while traceback:
        frame = traceback.tb_frame
        log.write(f"  File \"{frame.f_code.co_filename}\", line {traceback.tb_lineno}\n")
        traceback = traceback.tb_next
warnings.showwarning = warn_with_traceback

run_num = 3
run_combos = list(itertools.combinations(range(1, run_num + 1), 2))

whole_brain_mask = load_mni152_brain_mask()
brain_masker = NiftiMasker(whole_brain_mask, smoothing_fwhm=0, standardize=True)

def extract_roi_sphere(img, coords, radius=6):
    roi_masker = maskers.NiftiSpheresMasker([tuple(coords)], radius=radius)
    seed_time_series = roi_masker.fit_transform(img)
    if np.ma.is_masked(seed_time_series):
        print("Masked values found in extract_roi_sphere output")
        print(f"Number of masked values: {np.ma.count_masked(seed_time_series)}")
    return np.mean(seed_time_series, axis=1).reshape(-1, 1)

def make_psy_cov(runs, ss):
    temp_dir = f'{raw_dir}/{ss}/ses-01'
    cov_dir = f'{temp_dir}/covs'
    vols, tr = 184, 2.0
    times = np.arange(0, vols * len(runs) * tr, tr)
    full_cov = pd.DataFrame(columns=['onset', 'duration', 'value'])
    
    for rn, run in enumerate(runs):
        ss_num = ss.split('-')[1]
        obj_cov_file = f'{cov_dir}/catloc_{ss_num}_run-0{run}_Object.txt'
        scr_cov_file = f'{cov_dir}/catloc_{ss_num}_run-0{run}_Scramble.txt'
        
        if not os.path.exists(obj_cov_file) or not os.path.exists(scr_cov_file):
            print(f'Covariate file not found for run {run}')
            raise FileNotFoundError(f"Missing covariate file for run {run}")
        
        obj_cov = pd.read_csv(obj_cov_file, sep='\t', header=None, names=['onset', 'duration', 'value'])
        scr_cov = pd.read_csv(scr_cov_file, sep='\t', header=None, names=['onset', 'duration', 'value'])
        scr_cov['value'] *= -1
        
        curr_cov = pd.concat([obj_cov, scr_cov])
        curr_cov['onset'] += rn * vols * tr
        full_cov = pd.concat([full_cov, curr_cov])
    
    full_cov = full_cov.sort_values(by=['onset']).reset_index(drop=True)
    
    if full_cov['onset'].max() >= times[-1] or full_cov['onset'].min() < 0:
        raise ValueError("Event onsets are outside the expected time range")
    
    cov = full_cov.to_numpy()
    
    psy, _ = compute_regressor(cov.T, 'spm', times)
    
    if len(psy) != vols * len(runs):
        raise ValueError(f"Generated psy does not match expected length. Expected {vols * len(runs)}, got {len(psy)}")
    
    return psy

def calculate_fc(brain_time_series, phys):
    min_length = min(brain_time_series.shape[0], phys.shape[0])
    brain_time_series = brain_time_series[:min_length]
    phys = phys[:min_length].ravel()

    print(f"In calculate_fc:")
    print(f"  Brain time series shape: {brain_time_series.shape}")
    print(f"  Phys shape: {phys.shape}")

    constant_voxels = np.where(np.std(brain_time_series, axis=0) == 0)[0]
    print(f"  Number of constant voxels: {len(constant_voxels)}")
    if len(constant_voxels) > 0:
        print(f"  Indices of some constant voxels: {constant_voxels[:10]}...")

    correlations = []
    for i, voxel_ts in enumerate(brain_time_series.T):
        if np.std(voxel_ts) == 0:
            correlations.append(np.nan)
        else:
            corr = np.corrcoef(phys, voxel_ts)[0,1]
            correlations.append(corr)

        if i % 100000 == 0:
            print(f"  Processed {i} voxels")

    correlations = np.array(correlations)
    print(f"  Number of NaN correlations: {np.isnan(correlations).sum()}")
    return np.arctanh(correlations)

def check_roi_registration(roi_img, output_file):
    mni_template = datasets.load_mni152_template()
    
    display = plotting.plot_roi(roi_img, bg_img=mni_template, 
                                cut_coords=(0, 0, 0), display_mode='ortho',
                                title="ROI Registration Check")
    display.savefig(output_file)
    display.close()

def check_same_space(img1, img2):
    return np.allclose(img1.affine, img2.affine)

def conduct_analyses():
    start_time = time.time()
    for ss in subs:
        subject_start_time = time.time()
        print(f"Processing subject: {ss}")
        sub_dir = f'{study_dir}/{ss}/ses-01/'
        roi_dir = f'{sub_dir}/derivatives/rois'
        exp = 'loc'
        exp_dir = f'{sub_dir}/derivatives/fsl/{exp}'
        out_dir = os.path.join(sub_dir, 'derivatives', 'fc_ppi')
        os.makedirs(out_dir, exist_ok=True)

        roi_coords = pd.read_csv(f'{roi_dir}/spheres/sphere_coords_sandbox.csv')

        for rr in rois:
            all_ppi_runs = []
            all_fc_runs = []
            
            roi_path = f'{mni_parcel_dir}/{rr}.nii.gz'
            if os.path.exists(roi_path):
                roi_img = nib.load(roi_path)
            else:
                print(f"Standardized ROI file not found: {roi_path}")
                continue

            check_roi_registration(roi_img, os.path.join(out_dir, f'{ss}_{rr}_registration_check.png'))
            
            for rcn, rc in enumerate(run_combos):
                curr_coords = roi_coords[(roi_coords['index'] == rcn) & 
                                         (roi_coords['task'] == exp) & 
                                         (roi_coords['roi'] == rr)]
                
                if curr_coords.empty:
                    print(f"No coordinates found for {ss}, {rr}, {exp}, RC: {rc}")
                    continue

                peak_coords = curr_coords[['x', 'y', 'z']].values[0]

                held_out_run = list(set(range(1, 4)) - set(rc))[0]
                img4d = image.load_img(f'{exp_dir}/run-0{held_out_run}/1stLevel.feat/filtered_func_data_reg.nii.gz')
                
                if not check_same_space(roi_img, img4d):
                    print(f"Warning: ROI {rr} and functional data are not in the same space")

                print(f"Original img4d shape: {img4d.shape}")
                print(f"Any NaNs in original img4d: {np.isnan(img4d.get_fdata()).any()}")
                
                img4d = image.clean_img(img4d, standardize=True)
                print(f"Cleaned img4d shape: {img4d.shape}")
                print(f"Any NaNs in cleaned img4d: {np.isnan(img4d.get_fdata()).any()}")

                phys = extract_roi_sphere(img4d, peak_coords)
                print(f"Phys shape: {phys.shape}")
                print(f"Any NaNs in phys: {np.isnan(phys).any()}")

                psy = make_psy_cov([held_out_run], ss)
                print(f"Psy shape: {psy.shape}")
                print(f"Any NaNs in psy: {np.isnan(psy).any()}")

                brain_time_series = brain_masker.fit_transform(img4d)
                print(f"Brain time series shape: {brain_time_series.shape}")
                print(f"Any NaNs in brain time series: {np.isnan(brain_time_series).any()}")
                
                fc_correlations = calculate_fc(brain_time_series, phys)
                fc_img = brain_masker.inverse_transform(fc_correlations)
                all_fc_runs.append(fc_img)

                confounds = pd.DataFrame({'psy': psy[:,0], 'phys': phys[:,0]})
                ppi = psy * phys
                ppi = ppi.reshape((ppi.shape[0], 1))
                brain_time_series = brain_masker.fit_transform(img4d, confounds=confounds)
                seed_to_voxel_correlations = (np.dot(brain_time_series.T, ppi) / ppi.shape[0])
                seed_to_voxel_correlations = np.arctanh(seed_to_voxel_correlations)
                ppi_img = brain_masker.inverse_transform(seed_to_voxel_correlations.T)
                all_ppi_runs.append(ppi_img)

                print(f"{ss}, {rr}, {exp}, RC: {rc}, PPI max: {seed_to_voxel_correlations.max()}")

            mean_ppi = image.mean_img(all_ppi_runs)
            mean_fc = image.mean_img(all_fc_runs)
            
            nib.save(mean_ppi, os.path.join(out_dir, f'{ss}_{rr}_{exp}_ppi_sandbox.nii.gz'))
            nib.save(mean_fc, os.path.join(out_dir, f'{ss}_{rr}_{exp}_fc_sandbox.nii.gz'))
        
        subject_end_time = time.time()
        print(f"Finished processing subject {ss}. Time taken: {(subject_end_time - subject_start_time) / 60:.2f} minutes")
    
    end_time = time.time()
    print(f"Total processing time: {(end_time - start_time) / 3600:.2f} hours")

if __name__ == "__main__":
    conduct_analyses()