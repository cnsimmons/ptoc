import argparse
import pandas as pd
import numpy as np
import os
from nilearn import image, input_data
import nibabel as nib

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Run PPI analysis')
parser.add_argument('--sub', type=str, required=True, help='Subject ID')
parser.add_argument('--roi', type=str, required=True, help='ROI name')
args = parser.parse_args()

sub = args.sub
roi = args.roi

# Define paths and parameters
curr_dir = f'/user_data/csimmon2/git_repos/ptoc'
data_dir = '/path/to/data_dir'
results_dir = '/path/to/results_dir'
fig_dir = '/path/to/fig_dir'
raw_dir = '/path/to/raw_dir'
sub_info_path = f'{curr_dir}/sub_info.csv'
task_info_path = f'{curr_dir}/task_info.csv'
study = 'ptoc'
study_dir = f"/lab_data/behrmannlab/vlad/{study}"
ses = 1
suf = ''
tr = 2
vols = 184

# Ensure necessary directories exist
os.makedirs(f'{results_dir}/ppi', exist_ok=True)

# Load subject and task information
sub_info = pd.read_csv(sub_info_path)
task_info = pd.read_csv(task_info_path)

# Define additional necessary functions and variables

# Function to extract ROI sphere
def extract_roi_sphere(img, coords):
    roi_masker = input_data.NiftiSpheresMasker([tuple(coords)], radius=6)
    seed_time_series = roi_masker.fit_transform(img)
    phys = np.mean(seed_time_series, axis=1)
    phys = phys.reshape((phys.shape[0], 1))
    print(f'phys just ran')
    return phys

# Function to create the psychological regressor
def make_psy_cov(runs, ss):
    temp_dir = f'{raw_dir}/{ss}/ses-01'
    cov_dir = f'{temp_dir}/covs'
    
    times = np.arange(0, vols * tr, tr)
    full_cov = pd.DataFrame(columns=['onset', 'duration', 'value'])
    
    for rn, run in enumerate(runs):
        ss_num = ss.split('-')[1]
        curr_cov = pd.read_csv(f'{cov_dir}/catloc_{ss_num}_run-0{run}_Object.txt', sep='\t', header=None, names=['onset', 'duration', 'value'])
        curr_cont = pd.read_csv(f'{cov_dir}/catloc_{ss_num}_run-0{run}_Scramble.txt', sep='\t', header=None, names=['onset', 'duration', 'value'])
        curr_cont.iloc[:, 2] = curr_cont.iloc[:, 2] * -1
        curr_cov = curr_cov.append(curr_cont)
        full_cov = full_cov.append(curr_cov)
    
    full_cov = full_cov.sort_values(by=['onset'])
    cov = full_cov.to_numpy()
    psy, name = compute_regressor(cov.T, 'spm', times)
    
    print(f'Full covariate matrix shape: {cov.shape}')
    print(f'Created psy array shape: {psy.shape}')
    print(f'psy just ran')
    return psy

# Function to conduct PPI analysis
def conduct_ppi(sub, roi):
    sub_dir = f'{study_dir}/{sub}/ses-0{ses}'
    roi_dir = f'{sub_dir}/derivatives/rois'
    raw_dir = params.raw_dir
    temp_dir = f'{raw_dir}/{sub}/ses-01/derivatives/fsl/loc'
    
    roi_coords = pd.read_csv(f'{roi_dir}/spheres/sphere_coords.csv')
    out_dir = f'{study_dir}/{sub}/ses-01/derivatives/fc'
    os.makedirs(out_dir, exist_ok=True)
    print(f'Output directory ensured at {out_dir}')

    for tsk in ['loc']:
        all_runs = []
        for rcn, rc in enumerate(run_combos):
            curr_coords = roi_coords[(roi_coords['index'] == rcn) & (roi_coords['task'] == tsk) & (roi_coords['roi'] == roi)]
            for rn in rc:
                filtered_list = []
                curr_run = image.load_img(f'{temp_dir}/run-0{rn}/1stLevel.feat/filtered_func_data_reg.nii.gz')
                curr_run = image.clean_img(curr_run, standardize=True)
                filtered_list.append(curr_run)
            print('Loaded filtered data')
            
            img4d = image.concat_imgs(filtered_list)
            print('Loaded 4D image')
            
            phys = extract_roi_sphere(img4d, curr_coords[['x', 'y', 'z']].values.tolist()[0])
            print('Extracted sphere')
            
            psy = make_psy_cov([rn], sub)
            print('Loaded psy cov')
            
            print(f'Shape of img4d: {img4d.shape}')
            print(f'Length of phys: {phys.shape[0]}')
            print(f'Length of psy: {psy.shape[0]}')
            
            assert phys.shape[0] == psy.shape[0], f"Length mismatch: phys={phys.shape[0]}, psy={psy.shape[0]}"
            
            confounds = pd.DataFrame(columns=['psy', 'psy_lagged', 'phys', 'interact', 'constant'])
            confounds['psy'] = psy
            confounds['psy_lagged'] = confounds['psy'].shift()
            confounds['phys'] = phys
            confounds['interact'] = confounds['psy'] * confounds['phys']
            confounds['constant'] = np.ones((phys.shape[0], 1))

            print('Confounds dataframe:')
            print(confounds.head())
            
            confounds.to_csv(f'{out_dir}/{roi}_{tsk}_confounds.csv', index=False)
            print(f'Confounds saved at {out_dir}/{roi}_{tsk}_confounds.csv')

# Run the PPI analysis
conduct_ppi(sub, roi)