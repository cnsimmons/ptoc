#trying to improve efficiency of the code
import os
import pandas as pd
import numpy as np
from nilearn import image, input_data
from nilearn.maskers import NiftiMasker
from nilearn.datasets import load_mni152_brain_mask
from nilearn.glm.first_level import compute_regressor
import nibabel as nib
import sys

# Import your parameters
curr_dir = f'/user_data/csimmon2/git_repos/ptoc'
sys.path.insert(0, curr_dir)
import ptoc_params as params

# Set up directories and parameters
study = 'ptoc'
study_dir = f"/lab_data/behrmannlab/vlad/{study}"
results_dir = '/user_data/csimmon2/GitHub_Repos/ptoc/results'
raw_dir = params.raw_dir

sub_info = pd.read_csv(f'{curr_dir}/sub_info.csv')
subs = sub_info[sub_info['group'] == 'control']['sub'].tolist()
#subs = ['sub-107', 'sub-064', 'sub-025']
subs = ['sub-025']
rois = ['pIPS']
run_num = 3
runs = list(range(1, run_num + 1))
run_combos = [[rn1, rn2] for rn1 in range(1, run_num + 1) for rn2 in range(rn1 + 1, run_num + 1)]

whole_brain_mask = load_mni152_brain_mask()
brain_masker = NiftiMasker(whole_brain_mask, smoothing_fwhm=0, standardize=True)

def extract_roi_sphere(img, coords):
    roi_masker = input_data.NiftiSpheresMasker([tuple(coords)], radius=6)
    seed_time_series = roi_masker.fit_transform(img)
    phys = np.mean(seed_time_series, axis=1).reshape(-1, 1)
    return phys

def make_psy_cov(runs, ss):
    temp_dir = f'{raw_dir}/{ss}/ses-01'
    cov_dir = f'{temp_dir}/covs'
    vols, tr = 184, 2.0
    times = np.arange(0, vols * tr, tr)
    full_cov = pd.DataFrame(columns=['onset', 'duration', 'value'])

    for rn in runs:
        ss_num = ss.split('-')[1]
        obj_cov_file = f'{cov_dir}/catloc_{ss_num}_run-0{rn}_Object.txt'
        scr_cov_file = f'{cov_dir}/catloc_{ss_num}_run-0{rn}_Scramble.txt'

        if not os.path.exists(obj_cov_file) or not os.path.exists(scr_cov_file):
            print(f'Covariate file not found for run {rn}')
            continue

        obj_cov = pd.read_csv(obj_cov_file, sep='\t', header=None, names=['onset', 'duration', 'value'])
        scr_cov = pd.read_csv(scr_cov_file, sep='\t', header=None, names=['onset', 'duration', 'value'])
        scr_cov['value'] *= -1
        full_cov = pd.concat([full_cov, obj_cov, scr_cov])

    full_cov = full_cov.sort_values(by=['onset']).reset_index(drop=True)
    cov = full_cov.to_numpy()
    valid_onsets = cov[:, 0] < times[-1]
    cov = cov[valid_onsets]

    if cov.shape[0] == 0:
        print('No valid covariate data after filtering. Returning zeros array.')
        return np.zeros((vols, 1))

    psy, _ = compute_regressor(cov.T, 'spm', times)
    return psy

def conduct_analyses():
    for ss in subs:
        print(f"Processing subject: {ss}")
        sub_dir = f'{study_dir}/{ss}/ses-01/'
        roi_dir = f'{sub_dir}derivatives/rois'
        temp_dir = f'{raw_dir}/{ss}/ses-01/derivatives/fsl/loc'
        
        roi_coords = pd.read_csv(f'{roi_dir}/spheres/sphere_coords_sandbox.csv')
        
        out_dir = f'{study_dir}/{ss}/ses-01/derivatives'
        os.makedirs(f'{out_dir}/fc', exist_ok=True)
        os.makedirs(f'{out_dir}/fc', exist_ok=True)

        for tsk in ['loc']:
            for rr in rois:
                print(f"Processing ROI: {rr}")
                
                fc_file = f'{out_dir}/fc/{ss}_{rr}_{tsk}_fc_sandbox.nii.gz'
                ppi_file = f'{out_dir}/fc/{ss}_{rr}_{tsk}_ppi_sandbox.nii.gz'
                
                do_fc = not os.path.exists(fc_file)
                do_ppi = not os.path.exists(ppi_file)
                
                if not do_fc and not do_ppi:
                    print(f'Both FC and PPI files for {rr} already exist. Skipping...')
                    continue
                
                all_runs_fc = []
                all_runs_ppi = []
                
                for rcn, rc in enumerate(run_combos):
                    curr_coords = roi_coords[(roi_coords['index'] == rcn) & (roi_coords['task'] == tsk) & (roi_coords['roi'] == rr)]
                    coords = curr_coords[['x', 'y', 'z']].values.tolist()[0]
                    
                    filtered_list = [image.clean_img(image.load_img(f'{temp_dir}/run-0{rn}/1stLevel.feat/filtered_func_data_reg.nii.gz'), standardize=True) for rn in rc]
                    img4d = image.concat_imgs(filtered_list)
                    
                    phys = extract_roi_sphere(img4d, coords)
                    
                    # Ensure phys length matches the number of volumes
                    if phys.shape[0] > 184 * len(rc):
                        phys = phys[:184 * len(rc)]
                    
                    brain_time_series = brain_masker.fit_transform(img4d)
                    
                    if do_fc:
                        # FC Analysis
                        correlations = np.dot(brain_time_series.T, phys) / phys.shape[0]
                        correlations = np.arctanh(correlations.ravel())
                        correlation_img = brain_masker.inverse_transform(correlations)
                        all_runs_fc.append(correlation_img)
                    
                    if do_ppi:
                        # PPI Analysis
                        psy = make_psy_cov(rc, ss)  # Generate psy for the current run combination
                        
                        # Ensure psy length matches phys
                        if psy.shape[0] > phys.shape[0]:
                            psy = psy[:phys.shape[0]]
                        elif psy.shape[0] < phys.shape[0]:
                            phys = phys[:psy.shape[0]]
                            brain_time_series = brain_time_series[:psy.shape[0]]
                        
                        ppi_regressor = phys * psy
                        ppi_correlations = np.dot(brain_time_series.T, ppi_regressor) / ppi_regressor.shape[0]
                        ppi_correlations = np.arctanh(ppi_correlations.ravel())
                        ppi_img = brain_masker.inverse_transform(ppi_correlations)
                        all_runs_ppi.append(ppi_img)
                
                if do_fc:
                    mean_fc = image.mean_img(all_runs_fc)
                    nib.save(mean_fc, fc_file)
                    print(f'Saved FC result for {rr}')
                
                if do_ppi:
                    mean_ppi = image.mean_img(all_runs_ppi)
                    nib.save(mean_ppi, ppi_file)
                    print(f'Saved PPI result for {rr}')

# Call the function
conduct_analyses()