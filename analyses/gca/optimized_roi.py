import os
import pandas as pd
import numpy as np
from nilearn import image
from nilearn.input_data import NiftiSpheresMasker
import nibabel as nib
import sys

curr_dir = f'/user_data/csimmon2/git_repos/ptoc'
sys.path.insert(0, curr_dir)
import ptoc_params as params

# Set up directories and parameters
study = 'ptoc'
study_dir = f"/lab_data/behrmannlab/vlad/{study}"
results_dir = '/user_data/csimmon2/git_repos/ptoc/results'
raw_dir = params.raw_dir

##TO RUN ALL
sub_info = pd.read_csv(f'{curr_dir}/sub_info.csv')
sub_info = sub_info[sub_info['group'] == 'control']
subs = sub_info['sub'].tolist()
subs = [sub for sub in subs if sub != 'sub-025'] #all subs but 25

rois = ['pIPS', 'LO']
hemispheres = ['left', 'right']
run_num = 3

run_combos = [[rn1, rn2] for rn1 in range(1, run_num + 1) for rn2 in range(rn1 + 1, run_num + 1)]

def extract_roi_mean_activation(img, coords):
    roi_masker = NiftiSpheresMasker([tuple(coords)], radius=6)
    roi_time_series = roi_masker.fit_transform(img)
    return np.mean(roi_time_series)

def conduct_analyses():
    for ss in subs:
        print(f"Processing subject: {ss}")
        sub_dir = f'{study_dir}/{ss}/ses-01/'
        roi_dir = f'{sub_dir}derivatives/rois'
        temp_dir = f'{raw_dir}/{ss}/ses-01/derivatives/fsl/loc'
        
        roi_coords = pd.read_csv(f'{roi_dir}/spheres/sphere_coords_hemisphere.csv')
        
        out_dir = f'{study_dir}/{ss}/ses-01/derivatives'
        os.makedirs(f'{out_dir}/roi_activations', exist_ok=True)
        
        for tsk in ['loc']:
            for rr in rois:
                for hemi in hemispheres:
                    print(f"Processing ROI: {rr}, Hemisphere: {hemi}")
                    
                    roi_file = f'{out_dir}/roi_activations/{ss}_{rr}_{hemi}_{tsk}_roi_activation.csv'
                    
                    if os.path.exists(roi_file):
                        print(f"ROI activation file already exists for {rr} {hemi}. Skipping.")
                        continue
                    
                    all_runs_roi = []
                    
                    for rcn, rc in enumerate(run_combos):
                        curr_coords = roi_coords[(roi_coords['index'] == rcn) & 
                                                 (roi_coords['task'] == tsk) & 
                                                 (roi_coords['roi'] == rr) &
                                                 (roi_coords['hemisphere'] == hemi)]
                        
                        if curr_coords.empty:
                            print(f"No coordinates found for {rr}, {hemi}, run combo {rc}")
                            continue
                        
                        coords = curr_coords[['x', 'y', 'z']].values.tolist()[0]
                        
                        filtered_list = [image.load_img(f'{temp_dir}/run-0{rn}/1stLevel.feat/filtered_func_data_reg.nii.gz') for rn in rc]
                        img4d = image.concat_imgs(filtered_list)
                        
                        # ROI mean activation
                        roi_mean = extract_roi_mean_activation(img4d, coords)
                        all_runs_roi.append(roi_mean)

                    roi_df = pd.DataFrame({
                        'run_combo': [f'run_{rc[0]}_{rc[1]}' for rc in run_combos],
                        'mean_activation': all_runs_roi
                    })
                    roi_df.to_csv(roi_file, index=False)
                    print(f'Saved ROI mean activation for {rr} {hemi}')

# Call the function
conduct_analyses()