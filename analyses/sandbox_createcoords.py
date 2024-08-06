import os
import pandas as pd
import numpy as np
from nilearn import image, maskers, plotting
from nilearn.maskers import NiftiMasker
from nilearn.datasets import load_mni152_brain_mask,load_mni152_template
from nilearn import datasets
from nilearn.glm.first_level import compute_regressor
import nibabel as nib
import sys
import time
import itertools 

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

sub_info = pd.read_csv(f'{curr_dir}/sub_info.csv')
subs = sub_info[sub_info['group'] == 'control']['sub'].tolist()
rois = ['LO', 'pIPS']

'''scan params'''
tr = 2.0
vols = 184

whole_brain_mask = load_mni152_brain_mask()
mni = load_mni152_template()
brain_masker = NiftiMasker(whole_brain_mask, smoothing_fwhm=0, standardize=True)

#run combos
run_num = 3
runs = list(range(1,run_num+1))
run_combos = []

#determine the number of left out run combos
for rn1 in range(1,run_num+1):
    for rn2 in range(rn1+1,run_num+1):
        run_combos.append([rn1,rn2])

##def extract_roi_coords
def extract_roi_coords(): #define seed region in each roi | peak activation in ROI in first run, second run, etc. Saved in sphere coords, this one creates a loop for runs, so update to include the list runs. Creates a list to loop thorugh all possible combinations of runs
    """
    Define ROIs
    """
    parcels = ['LO', 'pIPS']

    for ss in subs:
        sub_dir = f'{study_dir}/{ss}/ses-01'
        roi_dir = f'{sub_dir}/derivatives/rois'
        os.makedirs(f'{roi_dir}/spheres', exist_ok=True)
        
        '''make roi spheres for spaceloc'''
        
        exp_dir = f'{sub_dir}/derivatives/fsl'
        parcel_dir = f'{roi_dir}/parcels'
        roi_coords = pd.DataFrame(columns = ['index','task','roi','x','y','z'])
        for rcn, rc in enumerate(run_combos): #determine which runs to use for creating ROIs
            roi_runs = [ele for ele in runs if ele not in rc]

            #load each run
            all_runs = []
            for rn in roi_runs:
                curr_run = image.load_img(f'{exp_dir}/loc/run-0{rn}/1stLevel_roi.feat/stats/zstat3_reg.nii.gz')
                all_runs.append(curr_run)

            mean_zstat = image.mean_img(all_runs)
            affine = mean_zstat.affine
            
            #loop through parcel determine coord of peak voxel
            for pr in parcels:

                #load parcel
                roi = image.load_img(f'{parcel_dir}/{lr}{pr}.nii.gz')
                roi = image.math_img('img > 0', img=roi)

                #masked_image = roi*image.get_data(mean_zstat)
                coords = plotting.find_xyz_cut_coords(mean_zstat,mask_img=roi, activation_threshold = .99) #99% threshold, i.e. highest 1% of voxels

                masked_stat = image.math_img('img1 * img2', img1=roi, img2=mean_zstat)
                masked_stat = image.get_data(masked_stat)
                np_coords = np.where(masked_stat == np.max(masked_stat))
                #max_coord = image.coord_transform(np_coords,affine)

                #masked_image = nib.Nifti1Image(masked_image, affine)  # create the volume image
                curr_coords = pd.Series([rcn, exp, f'{lr}{pr}'] + coords, index=roi_coords.columns)
                roi_coords = roi_coords.append(curr_coords,ignore_index = True)

        roi_coords.to_csv(f'{roi_dir}/spheres/sphere_coords.csv', index=False)