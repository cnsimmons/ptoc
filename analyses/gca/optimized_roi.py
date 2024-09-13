import os
import pandas as pd
import numpy as np
from nilearn import image, maskers
from nilearn.datasets import load_mni152_brain_mask
from nilearn.glm.first_level import compute_regressor
import nibabel as nib
import sys
from joblib import Parallel, delayed, Memory

# parameters
curr_dir = f'/user_data/csimmon2/git_repos/ptoc'
sys.path.insert(0, curr_dir)
import ptoc_params as params

study = 'ptoc'
study_dir = f"/lab_data/behrmannlab/vlad/{study}"
results_dir = '/user_data/csimmon2/git_repos/ptoc/results'
raw_dir = params.raw_dir

sub_info = pd.read_csv(f'{curr_dir}/sub_info.csv')
subs = sub_info[sub_info['group'] == 'control']['sub'].tolist()

rois = ['pIPS', 'LO']
hemispheres = ['left', 'right']
run_num = 3

runs = list(range(1, run_num + 1))
run_combos = [[rn1, rn2] for rn1 in range(1, run_num + 1) for rn2 in range(rn1 + 1, run_num + 1)]

# Set up caching
memory = Memory(cachedir='/tmp/nilearn_cache', verbose=0)

@memory.cache
def extract_roi_mean_activation(img, coords):
    roi_masker = maskers.NiftiSpheresMasker([tuple(coords)], radius=6, memory='nilearn_cache')
    roi_time_series = roi_masker.fit_transform(img)
    return np.mean(roi_time_series)

@memory.cache
def get_subject_mask(ss):
    mask_path = f'{raw_dir}/{ss}/ses-01/anat/{ss}_ses-01_T1w_brain_mask.nii.gz'
    return nib.load(mask_path)

def process_subject_roi(ss, rr, hemi, tsk):
    print(f"Processing subject: {ss}, ROI: {rr}, Hemisphere: {hemi}")
    sub_dir = f'{study_dir}/{ss}/ses-01/'
    roi_dir = f'{sub_dir}derivatives/rois'
    temp_dir = f'{raw_dir}/{ss}/ses-01/derivatives/fsl/loc'
    
    roi_coords = pd.read_csv(f'{roi_dir}/spheres/sphere_coords_hemisphere.csv')
    
    out_dir = f'{study_dir}/{ss}/ses-01/derivatives'
    os.makedirs(f'{out_dir}/fc', exist_ok=True)
    
    roi_file = f'{out_dir}/fc/{ss}_{rr}_{hemi}_{tsk}_roi.csv'
    
    if os.path.exists(roi_file):
        print(f"ROI file already exists for {ss}, {rr}, {hemi}")
        return
    
    whole_brain_mask = get_subject_mask(ss)
    
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
        
        # Use memory mapping for large files
        img_files = [f'{temp_dir}/run-0{rn}/1stLevel.feat/filtered_func_data_reg.nii.gz' for rn in rc]
        imgs = [image.load_img(img_file, mmap=True) for img_file in img_files]
        img4d = image.concat_imgs(imgs)
        
        # ROI mean activation
        roi_mean = extract_roi_mean_activation(img4d, coords)
        all_runs_roi.append(roi_mean)
    
    roi_df = pd.DataFrame({
        'run_combo': [f'run_{rc[0]}_{rc[1]}' for rc in run_combos],
        'mean_activation': all_runs_roi
    })
    roi_df.to_csv(roi_file, index=False)
    print(f'Saved ROI mean activation for {ss}, {rr} {hemi}')

def conduct_analyses():
    Parallel(n_jobs=-1)(delayed(process_subject_roi)(ss, rr, hemi, 'loc')
                        for ss in subs
                        for rr in rois
                        for hemi in hemispheres)

if __name__ == "__main__":
    conduct_analyses()