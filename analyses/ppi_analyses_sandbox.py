#this script for bash, the jupyter notebook is for testing

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
#from plotnine import ggplot, aes, geom_point

#hide warnings
import warnings
warnings.filterwarnings('ignore')

#load additional libraries
from nilearn import image, plotting, input_data, glm
from nilearn.input_data import NiftiMasker
import nibabel as nib
import statsmodels.api as sm
from nilearn.datasets import load_mni152_brain_mask, load_mni152_template


data_dir = params.data_dir
results_dir = params.results_dir
fig_dir = params.fig_dir
raw_dir = params.raw_dir

sub_info = params.sub_info
task_info = params.task_info

suf = params.suf
rois = params.rois
hemis = params.hemis

#load subject info
sub_info = pd.read_csv(f'{curr_dir}/sub_info.csv')

#mni = load_mni152_brain_mask()

'''exp info'''
subs = ['sub-064']  # Run for one subject initially
#subs = sub_info['sub'].tolist()

#Just controls
subs = sub_info[sub_info['group'] == 'control']['sub'].tolist()
study = 'ptoc'
data_dir = 'hemispace'
study_dir = f"/lab_data/behrmannlab/vlad/{study}"
out_dir = f'{study_dir}/derivatives/fc'
results_dir = '/user_data/csimmon2/GitHub_Repos/ptoc/results'
exp = ''
rois = ['LO']  # Run for one ROI initially
control_tasks = ['loc']
file_suf = ''

'''scan params'''
tr = 2
vols = 184

whole_brain_mask = load_mni152_brain_mask()
mni = load_mni152_template()
brain_masker = NiftiMasker(whole_brain_mask, smoothing_fwhm=0, standardize=True)

'''run info'''
run_num = 3
runs = list(range(1, run_num + 1))
run_combos = []

# Determine the number of left-out run combos
for rn1 in range(1, run_num + 1):
    for rn2 in range(rn1 + 1, run_num + 1):
        run_combos.append([rn1, rn2])
        

def extract_roi_coords():
    """
    Define ROIs
    """
    parcels = ['PPC', 'APC']

    for ss in subs:
        sub_dir = f'{study_dir}/{ss}/ses-01'
        roi_dir = f'{sub_dir}/derivatives/rois'
        os.makedirs(f'{roi_dir}/spheres', exist_ok=True)

        '''Make ROI spheres for loc'''

        exp_dir = f'{sub_dir}/derivatives/fsl/{exp}'
        parcel_dir = f'{roi_dir}/parcels'
        roi_coords = pd.DataFrame(columns=['index', 'task', 'roi', 'x', 'y', 'z'])
        for rcn, rc in enumerate(run_combos):  # Determine which runs to use for creating ROIs
            roi_runs = [ele for ele in runs if ele not in rc]

            # Load each run
            all_runs = []
            for rn in roi_runs:
                curr_run = image.load_img(f'{exp_dir}/run-0{rn}/1stLevel.feat/stats/zstat3_reg.nii.gz')

                all_runs.append(curr_run)

            mean_zstat = image.mean_img(all_runs)
            affine = mean_zstat.affine

            # Loop through parcel to determine coord of peak voxel
            for lr in ['l', 'r']:
                for pr in parcels:

                    # Load parcel
                    roi = image.load_img(f'{parcel_dir}/{pr}.nii.gz')
                    roi = image.math_img('img > 0', img=roi)

                    coords = plotting.find_xyz_cut_coords(mean_zstat, mask_img=roi, activation_threshold=.99)

                    masked_stat = image.math_img('img1 * img2', img1=roi, img2=mean_zstat)
                    masked_stat = image.get_data(masked_stat)
                    np_coords = np.where(masked_stat == np.max(masked_stat))

                    curr_coords = pd.Series([rcn, exp, f'{lr}{pr}'] + coords, index=roi_coords.columns)
                    roi_coords = roi_coords.append(curr_coords, ignore_index=True)

        roi_coords.to_csv(f'{roi_dir}/spheres/sphere_coords.csv', index=False)
#extract_roi_coords()


def extract_roi_sphere(img, coords):
    roi_masker = input_data.NiftiSpheresMasker([tuple(coords)], radius=6)
    seed_time_series = roi_masker.fit_transform(img)

    phys = np.mean(seed_time_series, axis=1)
    phys = phys.reshape((phys.shape[0], 1))

    return phys


def make_psy_cov(runs, ss):
    sub_dir = f'{study_dir}/{ss}/ses-01/'
    cov_dir = f'{sub_dir}/covs'
    times = np.arange(0, vols * len(runs), tr)
    full_cov = pd.DataFrame(columns=['onset', 'duration', 'value'])
    for rn, run in enumerate(runs):
        curr_cov = pd.read_csv(f'{cov_dir}/SpaceLoc_{study}{ss}_Run{run}_SA.txt', sep='\t', header=None, names=['onset', 'duration', 'value'])
        curr_cont = pd.read_csv(f'{cov_dir}/SpaceLoc_{study}{ss}_Run{run}_FT.txt', sep='\t', header=None, names=['onset', 'duration', 'value'])
        curr_cont.iloc[:, 2] = curr_cont.iloc[:, 2] * -1  # Make contrasting cov neg

        curr_cov = curr_cov.append(curr_cont)  # Append to positive

        curr_cov['onset'] = curr_cov['onset'] + (vols * rn)
        full_cov = full_cov.append(curr_cov)

    full_cov = full_cov.sort_values(by=['onset'])
    cov = full_cov.to_numpy()

    # Convolve to HRF
    psy, name = glm.first_level.compute_regressor(cov.T, 'spm', times)

    return psy


def conduct_ppi():
    for ss in subs:
        print(ss)
        sub_dir = f'{study_dir}/{ss}/ses-01/'
        cov_dir = f'{sub_dir}/covs'
        roi_dir = f'{sub_dir}/derivatives/rois'
        exp_dir = f'{sub_dir}/derivatives/fsl/{exp}'

        roi_coords = pd.read_csv(f'{roi_dir}/spheres/sphere_coords.csv')

        for tsk in ['loc']:
            for rr in dorsal_rois:
                all_runs = []  # This will get filled with the data from each run
                for rcn, rc in enumerate(run_combos):  # Determine which runs to use for creating ROIs
                    curr_coords = roi_coords[(roi_coords['index'] == rcn) & (roi_coords['task'] == tsk) & (roi_coords['roi'] == rr)]

                    filtered_list = []
                    for rn in rc:
                        curr_run = image.load_img(f'{exp_dir}/run-0{rn}/1stLevel.feat/filtered_func_data_reg.nii.gz')
                        curr_run = image.clean_img(curr_run, standardize=True)
                        filtered_list.append(curr_run)

                    img4d = image.concat_imgs(filtered_list)
                    phys = extract_roi_sphere(img4d, curr_coords[['x', 'y', 'z']].values.tolist()[0])
                    psy = make_psy_cov(rc, ss)

                    confounds = pd.DataFrame(columns=['psy', 'phys'])
                    confounds['psy'] = psy[:, 0]
                    confounds['phys'] = phys[:, 0]

                    ppi = psy * phys
                    ppi = ppi.reshape((ppi.shape[0], 1))

                    brain_time_series = brain_masker.fit_transform(img4d, confounds=[confounds])

                    seed_to_voxel_correlations = (np.dot(brain_time_series.T, ppi) / ppi.shape[0])
                    print(ss, rr, tsk, seed_to_voxel_correlations.max())

                    seed_to_voxel_correlations = np.arctanh(seed_to_voxel_correlations)
                    seed_to_voxel_correlations_img = brain_masker.inverse_transform(seed_to_voxel_correlations.T)

                    all_runs.append(seed_to_voxel_correlations_img)

                mean_fc = image.mean_img(all_runs)

                nib.save(mean_fc, f'{out_dir}/sub-{study}{ss}_{rr}_{tsk}_fc.nii.gz')

    
conduct_ppi()