import os
import pandas as pd
import numpy as np
from nilearn import image
from nilearn.input_data import NiftiSpheresMasker
import time
import sys

# parameters
curr_dir = f'/user_data/csimmon2/git_repos/ptoc'
sys.path.insert(0, curr_dir)
import ptoc_params as params

study = 'ptoc'
study_dir = f"/lab_data/behrmannlab/vlad/{study}"
results_dir = '/user_data/csimmon2/git_repos/ptoc/results'
raw_dir = params.raw_dir

#sub_info = pd.read_csv(f'{curr_dir}/sub_info.csv')
#subs = sub_info[sub_info['group'] == 'control']['sub'].tolist()
#subs = ['sub-038']

sub_info = pd.read_csv(f'{curr_dir}/sub_info.csv')
sub_info = sub_info[sub_info['group'] == 'control']
subs = sub_info['sub'].tolist()
subs = [sub for sub in subs if sub != 'sub-025'] #all subs but 25

rois = ['pIPS', 'LO']
hemispheres = ['left', 'right']
run_num = 3

run_combos = [[rn1, rn2] for rn1 in range(1, run_num + 1) for rn2 in range(rn1 + 1, run_num + 1)]

def load_and_concatenate_runs(temp_dir, run_num):
    start_time = time.time()
    all_run_data = []
    for run in range(1, run_num + 1):
        run_file = f'{temp_dir}/run-0{run}/1stLevel.feat/filtered_func_data_reg.nii.gz'
        if os.path.exists(run_file):
            all_run_data.append(image.load_img(run_file))
    img4d = image.concat_imgs(all_run_data) if all_run_data else None
    end_time = time.time()
    print(f"Time to load and concatenate runs: {end_time - start_time:.2f} seconds")
    return img4d

def extract_roi_mean_activation(img4d, coords):
    start_time = time.time()
    roi_masker = NiftiSpheresMasker([tuple(coords)], radius=6)
    roi_time_series = roi_masker.fit_transform(img4d)
    mean_activation = np.mean(roi_time_series)
    end_time = time.time()
    print(f"Time to extract ROI mean activation: {end_time - start_time:.2f} seconds")
    return mean_activation

def process_subject(ss):
    subject_start_time = time.time()
    print(f"Processing subject: {ss}")
    sub_dir = f'{study_dir}/{ss}/ses-01/'
    roi_dir = f'{sub_dir}derivatives/rois'
    temp_dir = f'{raw_dir}/{ss}/ses-01/derivatives/fsl/loc'
    
    roi_coords = pd.read_csv(f'{roi_dir}/spheres/sphere_coords_hemisphere.csv')
    
    out_dir = f'{study_dir}/{ss}/ses-01/derivatives/roi_activations'
    os.makedirs(out_dir, exist_ok=True)
    
    # Load and concatenate all runs once
    img4d = load_and_concatenate_runs(temp_dir, run_num)
    if img4d is None:
        print(f"No valid run data found for subject {ss}")
        return

    for tsk in ['loc']:
        for rr in rois:
            for hemi in hemispheres:
                roi_start_time = time.time()
                print(f"Processing ROI: {rr}, Hemisphere: {hemi}")
                
                roi_file = f'{out_dir}/{ss}_{rr}_{hemi}_{tsk}_roi_activation.csv'
                
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
                    
                    # Extract time series for this run combination
                    start_vol = (rc[0] - 1) * 184  # Assuming 184 volumes per run
                    end_vol = rc[1] * 184
                    run_img = image.index_img(img4d, slice(start_vol, end_vol))
                    
                    # ROI mean activation
                    roi_mean = extract_roi_mean_activation(run_img, coords)
                    all_runs_roi.append(roi_mean)

                roi_df = pd.DataFrame({
                    'run_combo': [f'run_{rc[0]}_{rc[1]}' for rc in run_combos],
                    'mean_activation': all_runs_roi
                })
                roi_df.to_csv(roi_file, index=False)
                roi_end_time = time.time()
                print(f'Saved ROI mean activation for {rr} {hemi}')
                print(f"Time to process ROI {rr} {hemi}: {roi_end_time - roi_start_time:.2f} seconds")

    subject_end_time = time.time()
    print(f"Total time to process subject {ss}: {subject_end_time - subject_start_time:.2f} seconds")

# Main execution
total_start_time = time.time()
for ss in subs:
    process_subject(ss)
total_end_time = time.time()
print(f"Total execution time: {total_end_time - total_start_time:.2f} seconds")
