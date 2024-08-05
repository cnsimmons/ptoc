import os
import pandas as pd
import numpy as np
from nilearn import image, input_data
from nilearn.maskers import NiftiMasker
from nilearn.datasets import load_mni152_brain_mask
from nilearn.glm.first_level import compute_regressor
import nibabel as nib
import sys
from multiprocessing import Pool

# Import your parameters
curr_dir = f'/user_data/csimmon2/git_repos/ptoc'
sys.path.insert(0, curr_dir)
import ptoc_params as params

# Set up directories and parameters
study = 'ptoc'
study_dir = f"/lab_data/behrmannlab/vlad/{study}"
results_dir = '/user_data/csimmon2/GitHub_Repos/ptoc/results'
raw_dir = params.raw_dir

#sub_info = pd.read_csv(f'{curr_dir}/sub_info.csv')
#subs = sub_info[sub_info['group'] == 'control']['sub'].tolist()
subs = ['sub-064']
rois = ['pIPS', 'LO', 'V1']
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

def conduct_ppi_analysis(img4d, phys, psy, brain_masker, coords):
    min_length = min(phys.shape[0], psy.shape[0])
    phys = phys[:min_length]
    psy = psy[:min_length]
    
    confounds = pd.DataFrame({'psy': psy.ravel(), 'phys': phys.ravel()})
    
    ppi = psy * phys
    
    brain_time_series = brain_masker.fit_transform(img4d, confounds=confounds)
    
    ppi_correlations = np.dot(brain_time_series.T, ppi) / ppi.shape[0]
    ppi_correlations = np.arctanh(ppi_correlations.ravel())
    
    ppi_img = brain_masker.inverse_transform(ppi_correlations)
    
    return ppi_img

def process_roi(args):
    ss, rr, tsk, rc, roi_coords, sub_dir, temp_dir, out_dir = args
    
    print(f"Processing subject: {ss}, ROI: {rr}, Task: {tsk}, Runs: {rc}")
    
    curr_coords = roi_coords[(roi_coords['index'] == rc[0]) & (roi_coords['task'] == tsk) & (roi_coords['roi'] == rr)]
    coords = curr_coords[['x', 'y', 'z']].values.tolist()[0]
    
    filtered_list = [image.clean_img(image.load_img(f'{temp_dir}/run-0{rn}/1stLevel.feat/filtered_func_data_reg.nii.gz'), standardize=True) for rn in rc]
    img4d = image.concat_imgs(filtered_list)
    
    phys = extract_roi_sphere(img4d, coords)
    
    if phys.shape[0] > 184 * len(rc):
        phys = phys[:184 * len(rc)]
    
    psy = make_psy_cov(rc, ss)
    
    if psy.shape[0] > phys.shape[0]:
        psy = psy[:phys.shape[0]]
    elif psy.shape[0] < phys.shape[0]:
        phys = phys[:psy.shape[0]]
    
    ppi_img = conduct_ppi_analysis(img4d, phys, psy, brain_masker, coords)
    
    return ppi_img

def conduct_analyses():
    for ss in subs:
        print(f"Processing subject: {ss}")
        sub_dir = f'{study_dir}/{ss}/ses-01/'
        roi_dir = f'{sub_dir}derivatives/rois'
        temp_dir = f'{raw_dir}/{ss}/ses-01/derivatives/fsl/loc'
        
        roi_coords = pd.read_csv(f'{roi_dir}/spheres/sphere_coords.csv')
        
        out_dir = f'{study_dir}/{ss}/ses-01/derivatives/fc'
        os.makedirs(out_dir, exist_ok=True)

        args_list = []
        for tsk in ['loc']:
            for rr in rois:
                ppi_file = f'{out_dir}/{ss}_{rr}_{tsk}_ppi_sandbox.nii.gz'
                
                if os.path.exists(ppi_file):
                    print(f'PPI file for {rr} already exists. Skipping...')
                    continue
                
                for rc in run_combos:
                    args_list.append((ss, rr, tsk, rc, roi_coords, sub_dir, temp_dir, out_dir))

        with Pool(processes=8) as pool:  # Adjust the number of processes as needed
            results = pool.map(process_roi, args_list)

        # Combine results
        for rr in rois:
            rr_results = [img for (subj, roi, tsk, rc, *_), img in zip(args_list, results) if roi == rr and subj == ss]
            if rr_results:
                mean_ppi = image.mean_img(rr_results)
                nib.save(mean_ppi, f'{out_dir}/{ss}_{rr}_{tsk}_ppi.nii.gz')
                print(f'Saved PPI result for {ss}, {rr}')

if __name__ == "__main__":
    conduct_analyses()