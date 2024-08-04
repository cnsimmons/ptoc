##FUNCTIONAL FC 
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nilearn.maskers import NiftiMasker, NiftiSpheresMasker
from nilearn import image, plotting, input_data
from nilearn.datasets import load_mni152_brain_mask, load_mni152_template
from nilearn.glm.first_level import compute_regressor
import nibabel as nib
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import scipy.stats as stats
import scipy
import statsmodels.api as s
from sklearn import metrics
from plotnine import *

curr_dir = f'/user_data/csimmon2/git_repos/ptoc'
sys.path.insert(0,curr_dir)
import ptoc_params as params

data_dir = params.data_dir
results_dir = params.results_dir
fig_dir = params.fig_dir
raw_dir = params.raw_dir
sub_info = params.sub_info
task_info = params.task_info

#sub_info = pd.read_csv(f'{curr_dir}/sub_info.csv')
#subs = sub_info[sub_info['group'] == 'control']['sub'].tolist()
subs = ['sub-038','sub-057','sub-059','sub-067'] #set 1
#subs = ['sub-068','sub-071','sub-083','sub-084'] #set 2
#subs = ['sub-085','sub-087','sub-088','sub-093'] #set 3
#subs = ['sub-094','sub-095','sub-096','sub-097'] #set 4
rois = ['LO','V1','aIPS']

study = 'ptoc'
study_dir = f"/lab_data/behrmannlab/vlad/{study}"
results_dir = '/user_data/csimmon2/GitHub_Repos/ptoc/results'
exp = ''
control_tasks = ['loc']
file_suf = ''

tr = 2
vols = 184

whole_brain_mask = load_mni152_brain_mask()
mni = load_mni152_template()
brain_masker = NiftiMasker(whole_brain_mask, smoothing_fwhm=0, standardize=True)

run_num = 3
runs = list(range(1, run_num + 1))
run_combos = [[rn1, rn2] for rn1 in range(1, run_num + 1) for rn2 in range(rn1 + 1, run_num + 1)]

def extract_roi_sphere(img, coords):
    roi_masker = input_data.NiftiSpheresMasker([tuple(coords)], radius = 6)
    seed_time_series = roi_masker.fit_transform(img)
    
    print(f"Extracted time series shape: {seed_time_series.shape}")
    print(f"Time series stats - Mean: {np.mean(seed_time_series)}, Std: {np.std(seed_time_series)}")
    
    phys = np.mean(seed_time_series, axis= 1)
    phys = phys.reshape((phys.shape[0],1))
    print(f"Phys regressor shape: {phys.shape}")
    print(f"Phys regressor stats - Mean: {np.mean(phys)}, Std: {np.std(phys)}")
    return phys

def make_psy_cov(runs, ss):
    temp_dir = f'{raw_dir}/{ss}/ses-01'
    cov_dir = f'{temp_dir}/covs'
    vols = 184  # Ensure this matches the expected number of volumes
    tr = 2.0  # Replace with the actual TR (Repetition Time) value in seconds
    times = np.arange(0, vols * tr, tr)
    full_cov = pd.DataFrame(columns=['onset', 'duration', 'value'])

    for rn in runs:
        ss_num = ss.split('-')[1]
        obj_cov_file = f'{cov_dir}/catloc_{ss_num}_run-0{rn}_Object.txt'
        scr_cov_file = f'{cov_dir}/catloc_{ss_num}_run-0{rn}_Scramble.txt'

        if not os.path.exists(obj_cov_file):
            print(f'Object covariate file not found: {obj_cov_file}')
            continue
        if not os.path.exists(scr_cov_file):
            print(f'Scramble covariate file not found: {scr_cov_file}')
            continue

        curr_cov = pd.read_csv(obj_cov_file, sep='\t', header=None, names=['onset', 'duration', 'value'])
        curr_cont = pd.read_csv(scr_cov_file, sep='\t', header=None, names=['onset', 'duration', 'value'])

        print(f'Object covariate file content (first 5 rows):\n{curr_cov.head()}')
        print(f'Scramble covariate file content (first 5 rows):\n{curr_cont.head()}')

        curr_cont.iloc[:, 2] *= -1
        curr_cov = pd.concat([curr_cov, curr_cont])
        full_cov = pd.concat([full_cov, curr_cov])

    full_cov = full_cov.sort_values(by=['onset']).reset_index(drop=True)
    print (full_cov)
    cov = full_cov.to_numpy()

    print(f'Full concatenated covariate data (first 10 rows):\n{full_cov.head(10)}')

    # Ensure the covariate timings are within the range of the times array
    valid_onsets = cov[:, 0] < times[-1]
    cov = cov[valid_onsets]

    if cov.shape[0] == 0:
        print('No valid covariate data after filtering. Returning zeros array.')
        return np.zeros((vols, 1))

    print(f'Covariate matrix for convolution:\n{cov}')

    psy, name = compute_regressor(cov.T, 'spm', times)
    return psy

def conduct_fc():
    for ss in subs:
        print(ss)
        sub_dir = f'{study_dir}/{ss}/ses-01/'
        roi_dir = f'{sub_dir}/derivatives/rois'
        raw_dir = params.raw_dir
        temp_dir = f'{raw_dir}/{ss}/ses-01/derivatives/fsl/loc'
        
        roi_coords = pd.read_csv(f'{roi_dir}/spheres/sphere_coords.csv')
        
        out_dir = f'{study_dir}/{ss}/ses-01/derivatives/fc'
        os.makedirs(out_dir, exist_ok=True)
        print(f'Output directory ensured at {out_dir}')

        for tsk in ['loc']:
            for rr in rois:
                print(f"Processing ROI: {rr}")
                
                fc_file = f'{out_dir}/{ss}_{rr}_{tsk}_fc.nii.gz'
                
                if os.path.exists(fc_file):
                    print(f'File {fc_file} already exists. Skipping...')
                    continue
                    
                all_runs_fc = []
                
                for rcn, rc in enumerate(run_combos):
                    curr_coords = roi_coords[(roi_coords['index'] == rcn) & (roi_coords['task'] == tsk) & (roi_coords['roi'] == rr)]
                    print(f"Using coordinates for ROI {rr}: {curr_coords[['x', 'y', 'z']].values.tolist()[0]}")
                    
                    for rn in rc:
                        filtered_list = []
                        curr_run = image.load_img(f'{temp_dir}/run-0{rn}/1stLevel.feat/filtered_func_data_reg.nii.gz') 
                        curr_run = image.clean_img(curr_run, standardize=True)
                        filtered_list.append(curr_run)    
                    print('Loaded filtered data')
                    
                    img4d = image.concat_imgs(filtered_list)
                    print('Loaded 4D image') 
                    
                    phys = extract_roi_sphere(img4d, curr_coords[['x', 'y', 'z']].values.tolist()[0])
                    print(f"Phys stats for {rr}: Mean = {np.mean(phys)}, Std = {np.std(phys)}")
                    
                    # Ensure phys length matches
                    if phys.shape[0] > 184:
                        phys = phys[:184]
                    
                    # FC SECTION
                    brain_time_series_fc = brain_masker.fit_transform(img4d)
                    print(f"FC brain time series stats for {rr}: Mean = {np.mean(brain_time_series_fc)}, Std = {np.std(brain_time_series_fc)}")

                    seed_to_voxel_correlations_fc = np.dot(brain_time_series_fc.T, phys) / phys.shape[0]
                    seed_to_voxel_correlations_fc = seed_to_voxel_correlations_fc.ravel()
                    print(f"FC correlation stats for {rr}: Min = {np.min(seed_to_voxel_correlations_fc)}, Max = {np.max(seed_to_voxel_correlations_fc)}")

                    seed_to_voxel_correlations_fc = np.arctanh(seed_to_voxel_correlations_fc)
                    print('Transformed FC correlation')

                    seed_to_voxel_correlations_img_fc = brain_masker.inverse_transform(seed_to_voxel_correlations_fc)
                    print(f"FC image stats for {rr}: Min = {np.min(seed_to_voxel_correlations_img_fc.get_fdata())}, Max = {np.max(seed_to_voxel_correlations_img_fc.get_fdata())}")

                    all_runs_fc.append(seed_to_voxel_correlations_img_fc)
                
                mean_fc = image.mean_img(all_runs_fc)
                print(f"FC mean image stats for {rr}: Min = {np.min(mean_fc.get_fdata())}, Max = {np.max(mean_fc.get_fdata())}")
                
                print(f"Saving FC file for {rr}: {fc_file}")
                nib.save(mean_fc, fc_file)
                print(f'Saved FC result: {fc_file}')

# Call the function
#conduct_ppi()
conduct_fc()