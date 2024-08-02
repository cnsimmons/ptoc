#run to start
curr_dir = f'/user_data/csimmon2/git_repos/ptoc'

import sys
sys.path.insert(0,curr_dir)
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import scipy.stats as stats
import scipy
import statsmodels.api as s
from sklearn import metrics

import pdb
import ptoc_params as params

from plotnine import *

#hide warnings
import warnings
warnings.filterwarnings('ignore')

#load additional libraries
from nilearn import image, plotting, input_data, glm
from nilearn.input_data import NiftiMasker
import nibabel as nib
import statsmodels.api as sm
from nilearn.datasets import load_mni152_brain_mask, load_mni152_template
from nilearn.glm.first_level import compute_regressor 

data_dir = params.data_dir
results_dir = params.results_dir
fig_dir = params.fig_dir
raw_dir = params.raw_dir
sub_info = params.sub_info
task_info = params.task_info

suf = params.suf
#mni = load_mni152_brain_mask()

'''exp info'''
#load subject info
sub_info = pd.read_csv(f'{curr_dir}/sub_info.csv')
subs = sub_info[sub_info['group'] == 'control']['sub'].tolist()

study = 'ptoc'
study_dir = f"/lab_data/behrmannlab/vlad/{study}"
results_dir = '/user_data/csimmon2/GitHub_Repos/ptoc/results'
exp = ''
control_tasks = ['loc']
file_suf = ''

'''scan params'''
tr = 2 #ptoc_params
vols = 184 #ptoc_params

whole_brain_mask = load_mni152_brain_mask()
mni = load_mni152_template()
brain_masker = NiftiMasker(whole_brain_mask, smoothing_fwhm=0, standardize=True)

#PPI and FC

'''run info'''
run_num =3
runs = list(range(1,run_num+1))
run_combos = []
#determine the number of left out run combos

for rn1 in range(1,run_num+1):
    for rn2 in range(rn1+1,run_num+1):
        run_combos.append([rn1,rn2])


#set1
rois = ['LO', 'aIPS', 'V1']
subs = ['sub-025','sub-038','sub-057','sub-059']

'''run info'''
run_num =3
runs = list(range(1,run_num+1))
run_combos = []
#determine the number of left out run combos

for rn1 in range(1,run_num+1):
    for rn2 in range(rn1+1,run_num+1):
        run_combos.append([rn1,rn2])

#phys
def extract_roi_sphere(img, coords):
    roi_masker = input_data.NiftiSpheresMasker([tuple(coords)], radius = 6)
    seed_time_series = roi_masker.fit_transform(img)
    
    phys = np.mean(seed_time_series, axis= 1)
    phys = phys.reshape((phys.shape[0],1))
    print (f'phys just ran')
    return phys

#psy
def make_psy_cov(runs, ss):
    temp_dir = f'{raw_dir}/{ss}/ses-01'
    cov_dir = f'{temp_dir}/covs'
    
    # Only for a single run
    times = np.arange(0, vols * tr, tr)  # Create time array covering the whole run duration
    full_cov = pd.DataFrame(columns=['onset', 'duration', 'value'])
    
    for rn, run in enumerate(runs):
        ss_num = ss.split('-')[1]  # Strips the "sub-" from the subject number
        curr_cov = pd.read_csv(f'{cov_dir}/catloc_{ss_num}_run-0{run}_Object.txt', sep='\t', header=None, names=['onset', 'duration', 'value'])
        
        # Contrasting (negative) covariate
        curr_cont = pd.read_csv(f'{cov_dir}/catloc_{ss_num}_run-0{run}_Scramble.txt', sep='\t', header=None, names=['onset', 'duration', 'value'])
        curr_cont.iloc[:, 2] = curr_cont.iloc[:, 2] * -1  # Make contrasting cov negative
        
        curr_cov = curr_cov.append(curr_cont)  # Append to positive
        
        # Append to concatenated cov
        full_cov = full_cov.append(curr_cov)
    
    full_cov = full_cov.sort_values(by=['onset'])
    cov = full_cov.to_numpy()

    # Convolve to HRF
    psy, name = compute_regressor(cov.T, 'spm', times)
    
    # Debug: Print the shape of the created psy array
    print(f'Full covariate matrix shape: {cov.shape}')
    print(f'Created psy array shape: {psy.shape}')
    print (f'psy just ran')
    return psy

#ppi
def conduct_ppi():
    for ss in subs:
        print(ss)
        sub_dir = f'{study_dir}/{ss}/ses-01/'  # study is PTOC
        roi_dir = f'{sub_dir}/derivatives/rois'  # rois in PTOC
        raw_dir = params.raw_dir  # hemispace
        temp_dir = f'{raw_dir}/{ss}/ses-01/derivatives/fsl/loc' # hemispace
        
        roi_coords = pd.read_csv(f'{roi_dir}/spheres/sphere_coords.csv')  # load ROI coordinates
        
        # Ensure output directory exists
        out_dir = f'{study_dir}/{ss}/ses-01/derivatives/fc'
        os.makedirs(out_dir, exist_ok=True)
        print(f'Output directory ensured at {out_dir}')

        for tsk in ['loc']:
            for rr in rois:
                
                ppi_file = f'{out_dir}/{ss}_{rr}_{tsk}_ppi.nii.gz'
                fc_file = f'{out_dir}/{ss}_{rr}_{tsk}_fc.nii.gz'
                
                if os.path.exists(ppi_file) and os.path.exists(fc_file):
                    print(f'Files {ppi_file} and {fc_file} already exist. Skipping...')
                    continue
                    
                all_runs_ppi = []
                all_runs_fc = []
                for rcn, rc in enumerate(run_combos):  # run combos
                    curr_coords = roi_coords[(roi_coords['index'] == rcn) & (roi_coords['task'] == tsk) & (roi_coords['roi'] == rr)]
                    for rn in rc:
                        filtered_list = []
                        curr_run = image.load_img(f'{temp_dir}/run-0{rn}/1stLevel.feat/filtered_func_data_reg.nii.gz') 
                        curr_run = image.clean_img(curr_run, standardize=True)
                        filtered_list.append(curr_run)    
                    print('Loaded filtered data')
                    
                    img4d = image.concat_imgs(filtered_list)
                    print('Loaded 4D image') 
                    
                    phys = extract_roi_sphere(img4d, curr_coords[['x', 'y', 'z']].values.tolist()[0])  # extract ROI sphere coordinate
                    print('Extracted sphere') 
                    
                    # Load behavioral data
                    psy = make_psy_cov([rn], ss)  # grabs the covariate and converts the three-column into binary data
                    print('Loaded psy cov')
                    
                    # Debug: Print shapes of img4d and the concatenated phys and psy arrays
                    print(f'Shape of img4d: {img4d.shape}')
                    print(f'Length of phys: {phys.shape[0]}')
                    print(f'Length of psy: {psy.shape[0]}')
                    
                    # Ensure phys and psy lengths match
                    assert phys.shape[0] == psy.shape[0], f"Length mismatch: phys={phys.shape[0]}, psy={psy.shape[0]}"
                    
                    # Combine phys (seed TS) and psy (task TS) into a regressor
                    confounds = pd.DataFrame(columns=['psy', 'phys'])
                    confounds['psy'] = psy[:, 0]
                    confounds['phys'] = phys[:, 0]
                    print('Combined psy and phys')

                    # Create PPI cov by multiplying psy * phys
                    ppi = psy * phys
                    ppi = ppi.reshape((ppi.shape[0], 1))
                    print('Created PPI')
                    
                    # Extract brain time series using PPI confounds
                    brain_time_series_ppi = brain_masker.fit_transform(img4d, confounds=[confounds])
                    print('Extracted brain TS for PPI')

                    # Correlate interaction term to TS for voxels in the brain for PPI
                    seed_to_voxel_correlations_ppi = (np.dot(brain_time_series_ppi.T, ppi) / ppi.shape[0])
                    print(ss, rr, tsk, seed_to_voxel_correlations_ppi.max())
                    print('Correlated interaction term for PPI')

                    # Transform PPI correlation back to brain space
                    seed_to_voxel_correlations_ppi = np.arctanh(seed_to_voxel_correlations_ppi)
                    print('Transformed PPI correlation')

                    # Transform PPI correlation map back to brain
                    seed_to_voxel_correlations_img_ppi = brain_masker.inverse_transform(seed_to_voxel_correlations_ppi.T)
                    print('Transformed PPI correlation map')

                    all_runs_ppi.append(seed_to_voxel_correlations_img_ppi)
                    
                    ##FC SECTION
                    # Extract brain time series using only psy confound for FC
                    brain_time_series_fc = brain_masker.fit_transform(img4d)
                    print('Extracted brain TS for FC')

                    # Correlate psy term for FC
                    seed_to_voxel_correlations_fc = (np.dot(brain_time_series_fc.T, psy) / psy.shape[0])
                    print('Correlated psy term for FC')

                    # Transform FC correlation back to brain space
                    seed_to_voxel_correlations_fc = np.arctanh(seed_to_voxel_correlations_fc)
                    print('Transformed FC correlation')

                    # Transform FC correlation map back to brain
                    seed_to_voxel_correlations_img_fc = brain_masker.inverse_transform(seed_to_voxel_correlations_fc.T)
                    print('Transformed FC correlation map')

                    all_runs_fc.append(seed_to_voxel_correlations_img_fc)

                mean_ppi = image.mean_img(all_runs_ppi)
                mean_fc = image.mean_img(all_runs_fc)
                
                # Save PPI results
                nib.save(mean_ppi, ppi_file)
                print(f'Saved PPI result: {ppi_file}')
                
                # Save FC results
                nib.save(mean_fc, fc_file)
                print(f'Saved FC result: {fc_file}')
conduct_ppi()
