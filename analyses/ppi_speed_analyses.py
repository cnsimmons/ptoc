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
subs = ['sub-097']
#subs = sub_info[sub_info['group'] == 'control']['sub'].tolist()

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

#PPI
#rois = params.rois
#rois = ['LO', 'PFS', 'pIPS', 'aIPS', 'V1']
rois = ['LO']

'''run info'''
run_num =3
runs = list(range(1,run_num+1))
run_combos = []
#determine the number of left out run combos

for rn1 in range(1,run_num+1):
    for rn2 in range(rn1+1,run_num+1):
        run_combos.append([rn1,rn2])
          
#create ROI Coords
def extract_roi_coords():
    """
    Define ROIs
    """
    parcels = ['V1', 'aIPS', 'PFS', 'pIPS', 'LO']
    #subs = sub_info[sub_info['group'] == 'control']['sub'].tolist()
    subs = ['sub-057']
    
    '''run info'''
    run_num =3
    runs = list(range(1,run_num+1))
    run_combos = []
    #determine the number of left out run combos

    for rn1 in range(1,run_num+1):
        for rn2 in range(rn1+1,run_num+1):
            run_combos.append([rn1,rn2])

    for ss in subs:
        print(f'Processing subject: {ss}')
        sub_dir = f'{study_dir}/{ss}/ses-01'
        roi_dir = f'{sub_dir}/derivatives/rois'
        os.makedirs(f'{roi_dir}/spheres', exist_ok=True)
        
        exp_dir = f'{sub_dir}/derivatives/fsl'
        parcel_dir = f'{roi_dir}/parcels'
        roi_coords = pd.DataFrame(columns=['index', 'task', 'roi', 'x', 'y', 'z'])
        
        for rcn, rc in enumerate(run_combos):
            roi_runs = [ele for ele in runs if ele not in rc]
            
            #load each run
            all_runs = []
            for rn in roi_runs:
                curr_run_path = f'{exp_dir}/loc/run-0{rn}/1stLevel.feat/stats/zstat3_reg.nii.gz'
                if os.path.exists(curr_run_path):
                    curr_run = image.load_img(curr_run_path)
                    all_runs.append(curr_run)
                else:
                    print(f'File does not exist: {curr_run_path}')
            
            mean_zstat = image.mean_img(all_runs)
            affine = mean_zstat.affine

            #loop through parcel determine coord of peak voxel
            for pr in parcels:
                
                roi_path = f'{parcel_dir}/{pr}.nii.gz'
                if os.path.exists(roi_path):
                    roi = image.load_img(roi_path)
                    roi = image.math_img('img > 0', img=roi)

                    coords = plotting.find_xyz_cut_coords(mean_zstat, mask_img=roi, activation_threshold=0.99)
                    
                    masked_stat = image.math_img('img1 * img2', img1=roi, img2=mean_zstat)
                    masked_stat = image.get_data(masked_stat)
                    np_coords = np.where(masked_stat == np.max(masked_stat))
                    
                    curr_coords = pd.Series([rcn, 'loc', pr] + coords, index=roi_coords.columns)
                    roi_coords = roi_coords.append(curr_coords, ignore_index=True)

                    # control task ROI
                    control_zstat_path = f'{exp_dir}/loc/HighLevel.gfeat/cope3.feat/stats/zstat1.nii.gz'
                    if os.path.exists(control_zstat_path):
                        control_zstat = image.load_img(control_zstat_path)
                        coords = plotting.find_xyz_cut_coords(control_zstat, mask_img=roi, activation_threshold=0.99)
                        
                        curr_coords = pd.Series([rcn, 'highlevel', pr] + coords, index=roi_coords.columns)
                        roi_coords = roi_coords.append(curr_coords, ignore_index=True)
                    else:
                        print(f'File does not exist: {control_zstat_path}')
                else:
                    print(f'File does not exist: {roi_path}')
            
        roi_coords.to_csv(f'{roi_dir}/spheres/sphere_coords.csv', index=False)

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
        temp_dir = f'{raw_dir}/{ss}/ses-01/derivatives/fsl/loc'  # hemispace
        
        roi_coords = pd.read_csv(f'{roi_dir}/spheres/sphere_coords.csv')  # load ROI coordinates
        
        # Ensure output directory exists
        out_dir = f'{study_dir}/{ss}/ses-01/derivatives/fc'
        os.makedirs(out_dir, exist_ok=True)
        print(f'Output directory ensured at {out_dir}')

        for tsk in ['loc']:
            for rr in rois:
                # Check if the file already exists
                output_file = f'{out_dir}/{ss}_{rr}_{tsk}_ppi.nii.gz'
                if os.path.exists(output_file):
                    print(f"Output file {output_file} already exists. Skipping analysis.")
                    continue
                        
                all_runs = []
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

                    # Extract brain time series with confounds removal
                    brain_time_series = brain_masker.fit_transform(img4d, confounds=[confounds])
                    print('Extracted brain TS with confounds removal')

                    # If needed, you can create a version without confounds removal here (if required)
                    # brain_time_series_4FC = brain_masker.fit_transform(img4d)  # This line is not required if the next steps do not use it

                    # Correlate interaction term to TS for vox in the brain
                    seed_to_voxel_correlations = (np.dot(brain_time_series.T, ppi) / ppi.shape[0])
                    print(ss, rr, tsk, seed_to_voxel_correlations.max())
                    print('Correlated interaction term')

                    # Correlate psy term
                    seed_to_voxel_correlations = (np.dot(brain_time_series.T, psy) / psy.shape[0])
                    print('Correlated psy term')

                    # Transform correlation back to brain space
                    seed_to_voxel_correlations = np.arctanh(seed_to_voxel_correlations)
                    print('Transformed correlation')

                    # Transform correlation map back to brain
                    seed_to_voxel_correlations_img = brain_masker.inverse_transform(seed_to_voxel_correlations.T)
                    print('Transformed correlation map')

                    all_runs.append(seed_to_voxel_correlations_img)

                mean_fc = image.mean_img(all_runs)
                
                # Ensure output directory exists
                out_dir = f'{study_dir}/{ss}/ses-01/derivatives/fc'
                os.makedirs(out_dir, exist_ok=True)
                print(f'Output directory ensured at {out_dir}')
                
                # Save the result
                nib.save(mean_fc, f'{out_dir}/{ss}_{rr}_{tsk}_ppi.nii.gz')
                print(f'Saved PPI to {out_dir}/{ss}_{rr}_{tsk}_ppi.nii.gz')
                nib.save(mean_fc, f'{out_dir}/{ss}_{rr}_{tsk}_fc_4FC.nii.gz')
                print(f'Saved FC to {out_dir}/{ss}_{rr}_{tsk}_fc_4FC.nii.gz')

#extract_roi_coords()
conduct_ppi()