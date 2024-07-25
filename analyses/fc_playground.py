curr_dir = f'/user_data/csimmon2/git_repos/ptoc'


import sys
sys.path.append(curr_dir)
import os
import pandas as pd
import numpy as np
from nilearn import image, input_data, plotting
from nilearn.datasets import load_mni152_brain_mask, load_mni152_template
import nibabel as nib
import warnings
import ptoc_params as params

# Other imports
from nilearn.input_data import NiftiMasker ### "is deprecated in version 0.9. Please import from nilearn.maskers instead"

warnings.filterwarnings('ignore')

# Study parameters
study = 'ptoc'
exp = 'loc'
file_suf = ''
sub_info = params.sub_info

data_dir = params.data_dir
results_dir = params.results_dir
fig_dir = params.fig_dir
task_info = params.task_info
thresh = params.thresh

suf = params.suf
rois = params.rois
#rois = ['LO', 'PFS', 'pIPS', 'aIPS']
#hemis = params.hemis
cope = params.cope

# Load MNI templates and masks
whole_brain_mask = load_mni152_brain_mask()
mni = load_mni152_template()
brain_masker = input_data.NiftiMasker(whole_brain_mask, smoothing_fwhm=0, standardize=True)


# Run information
run_num = 3
runs = list(range(1, run_num + 1))
run_combos = [(rn1, rn2) for rn1 in range(1, run_num) for rn2 in range(rn1 + 1, run_num + 1)]


# Define ROI names
parcels = ['LO', 'PFS', 'pIPS', 'aIPS', 'ventral_visual_cortex', 'dorsal_visual_cortex']

#parcels = ['lLO', 'lPFS', 'lpIPS', 'laIPS', 'lventral_visual_cortex', 'ldorsal_visual_cortex',
 #          'rLO', 'rPFS', 'rpIPS', 'raIPS', 'rventral_visual_cortex', 'rdorsal_visual_cortex']
#ss = sub_info['sub'][5] #2 = sub-090, 

# Function to extract ROI coordinates
def extract_roi_coords():
    for ss in sub_info['sub']:
        print(ss)
        #sub_dir = f'{study_dir}/sub-{study}{ss}/ses-01'
        sub_dir = f'{data_dir}/{ss}/ses-01'
        roi_dir = f'{sub_dir}/derivatives/rois'
        os.makedirs(f'{roi_dir}/spheres', exist_ok=True)
        
        exp_dir = f'{sub_dir}/derivatives/fsl'
        parcel_dir = f'{roi_dir}/parcels'
        roi_coords = pd.DataFrame(columns=['index', 'task', 'roi', 'x', 'y', 'z'])  # Create empty DataFrame
        for rcn, rc in enumerate(run_combos):

            all_runs = []
            for rn in rc:
                curr_run = image.load_img(
                    f'{exp_dir}/{exp}/run-0{rn}/1stLevel.feat/stats/zstat{cope}_reg.nii.gz')
                all_runs.append(curr_run)

            mean_zstat = image.mean_img(all_runs)
            affine = mean_zstat.affine

            for lr in ['l', 'r']:
                for pr in parcels:

                    roi = image.load_img(f'{parcel_dir}/{pr}.nii.gz')
                    #roi = image.load_img(f'{parcel_dir}/{lr}{pr}.nii.gz')
                    roi = image.math_img('img > 0', img=roi)

                    coords = plotting.find_xyz_cut_coords(mean_zstat, mask_img=roi, activation_threshold=0.99)

                    masked_stat = image.math_img('img1 * img2', img1=roi, img2=mean_zstat)
                    masked_stat = image.get_data(masked_stat)
                    np_coords = np.where(masked_stat == np.max(masked_stat))

                    curr_coords = pd.Series([rcn, exp, f'{lr}{pr}'] + coords, index=roi_coords.columns)
                    roi_coords = roi_coords.append(curr_coords, ignore_index=True)
        
        roi_coords.to_csv(f'{roi_dir}/spheres/sphere_coords.csv', index=False)
        

# Function to extract ROI sphere
def extract_roi_sphere(img, coords):
    roi_masker = input_data.NiftiSpheresMasker([tuple(coords)], radius=6)
    seed_time_series = roi_masker.fit_transform(img)
    
    phys = np.mean(seed_time_series, axis=1).reshape((-1, 1))
    return phys


# Function to conduct functional connectivity
def conduct_fc():

    #for each subject, load their individual anatomy and mask

    for sub, group, hemi in zip(sub_info['sub'], sub_info['group'], sub_info['intact_hemi']):
        print(sub)
        sub_dir = f'{data_dir}/{sub}/ses-01'
        cov_dir = f'{sub_dir}/covs'
        roi_dir = f'{sub_dir}/derivatives/rois'
        exp_dir = f'{sub_dir}/derivatives/fsl/{exp}'
        
        if hemi == 'both':
            hemis = ['l','r']
        elif hemi == 'left':
            hemis = ['l']
        elif hemi == 'right':
            hemis = ['r']
        

        roi_coords = pd.read_csv(f'{roi_dir}/spheres/sphere_coords.csv')

        for tsk in ['loc']: #change to just loc
            for lr in hemis:
                if lr == 'l':
                    curr_hemi = 'left'
                elif lr == 'r':
                    curr_hemi = 'right'
                #load anat mask
                anat_mask = image.load_img(f'{params.raw_dir}/{sub}/ses-01/anat/{sub}_ses-01_T1w_brain_mask_{curr_hemi}.nii.gz')
                #binarize mask
                anat_mask = image.math_img('img > 0', img=anat_mask)
                brain_masker = input_data.NiftiMasker(anat_mask, smoothing_fwhm=0, standardize=True)
                
                
                for rr in rois: #change to just rois
                    roi = f'{lr}{rr}'
                    all_runs = [] #this will get filled with the data from each run
                    for rcn, rc in enumerate(run_combos): #determine which runs to use for creating ROIs
                        curr_coords = roi_coords[(roi_coords['index'] == rcn) & (roi_coords['task'] ==tsk) & (roi_coords['roi'] ==roi)]

                        #Determine which run was not used to create the ROI
                        test_runs = [elem for elem in runs if elem not in rc ]

                        filtered_list = []
                        for rn in test_runs:
                            #extract roi time series dat afrom held out run
                            curr_run = image.load_img(f'{exp_dir}/run-0{rn}/1stLevel.feat/filtered_func_data_reg.nii.gz')
                            curr_run = image.clean_img(curr_run,standardize=True)
                            filtered_list.append(curr_run)
    
                        img4d = image.concat_imgs(filtered_list)
                        
                        #extract time series from peak voxel
                        if not curr_coords.empty:
                            phys = extract_roi_sphere(img4d, curr_coords[['x', 'y', 'z']].values.tolist()[0])
                        
                            #phys = extract_roi_sphere(img4d,curr_coords[['x','y','z']].values.tolist()[0])

                            #Extract time series from rest of brain
                            brain_time_series = brain_masker.fit_transform(img4d)

                            #Correlate interaction term to TS for vox in the brain
                            
                            seed_to_voxel_correlations = (np.dot(brain_time_series.T, phys) /
                                            phys.shape[0])
                            print(sub, roi, tsk, seed_to_voxel_correlations.max())
                            
                            #convert correlations to fisher z
                            seed_to_voxel_correlations = np.arctanh(seed_to_voxel_correlations)

                            #transform correlation map back to brain space
                            seed_to_voxel_correlations_img = brain_masker.inverse_transform(seed_to_voxel_correlations.T)
                            
                            all_runs.append(seed_to_voxel_correlations_img)
                        else:
                                print(f"No matching coordinates found for rcn={rcn}, tsk={tsk}, roi={roi}")

                    mean_fc = image.mean_img(all_runs)
                        
                    nib.save(mean_fc, f'{results_dir}/sub-{study}{sub}_{roi}_{tsk}_fc.nii.gz')

# Call functions
extract_roi_coords()  # Run this first
#conduct_fc()  # Run this after extracting ROI coordinates
