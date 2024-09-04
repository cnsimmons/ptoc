import os
import pandas as pd
import numpy as np
from nilearn import image, input_data
from statsmodels.tsa.stattools import grangercausalitytests
import sys
import nibabel as nib

# Import your parameters
curr_dir = f'/user_data/csimmon2/git_repos/ptoc'
sys.path.insert(0, curr_dir)
import ptoc_params as params

# Set up directories and parameters
study = 'ptoc'
study_dir = f"/lab_data/behrmannlab/vlad/{study}"
results_dir = '/user_data/csimmon2/git_repos/ptoc/results'
raw_dir = params.raw_dir
sub_info = pd.read_csv(f'{curr_dir}/sub_info.csv')
subs = ['sub-038']  # Update this list as needed
rois = ['pIPS', 'LO']  # We'll analyze the relationship between these two ROIs
hemispheres = ['left', 'right']
run_num = 3
runs = list(range(1, run_num + 1))
run_combos = [[rn1, rn2] for rn1 in range(1, run_num + 1) for rn2 in range(rn1 + 1, run_num + 1)]

def extract_roi_sphere(img, coords):
    #roi_masker = input_data.NiftiSpheresMasker([tuple(coords)], radius=6)
    roi_masker = input_data.NiftiSpheresMasker([tuple(coords)], radius=6, standardize=False, memory='nilearn_cache', memory_level=1)
    seed_time_series = roi_masker.fit_transform(img)
    return np.mean(seed_time_series, axis=1).reshape(-1, 1)

def extract_condition_timeseries(runs, ss):
    temp_dir = f'{raw_dir}/{ss}/ses-01'
    cov_dir = f'{temp_dir}/covs'
    vols, tr = 184, 2.0
    times = np.arange(0, vols * len(runs) * tr, tr)
    
    object_timeseries = np.zeros(len(times))
    
    for rn in runs:
        ss_num = ss.split('-')[1]
        obj_cov_file = f'{cov_dir}/catloc_{ss_num}_run-0{rn}_Object.txt'
        
        if not os.path.exists(obj_cov_file):
            print(f'Covariate file not found for run {rn}')
            continue
        
        obj_cov = pd.read_csv(obj_cov_file, sep='\t', header=None, names=['onset', 'duration', 'value'])
        
        # Adjust onsets for current run
        obj_cov['onset'] += (rn - 1) * vols * tr
        
        # Create binary timeseries for object condition
        for _, row in obj_cov.iterrows():
            start = int(row['onset'] / tr)
            end = int((row['onset'] + row['duration']) / tr)
            object_timeseries[start:end] = 1
    
    return object_timeseries

def conduct_gca(roi1_ts, roi2_ts, condition_timeseries):
    # Ensure all timeseries have the same length
    min_length = min(roi1_ts.shape[0], roi2_ts.shape[0], len(condition_timeseries))
    roi1_ts = roi1_ts[:min_length]
    roi2_ts = roi2_ts[:min_length]
    condition_timeseries = condition_timeseries[:min_length]
    
    # Extract condition-specific timeseries
    condition_mask = condition_timeseries == 1
    roi1_condition = roi1_ts[condition_mask]
    roi2_condition = roi2_ts[condition_mask]
    
    neural_ts = pd.DataFrame({'roi1': np.squeeze(roi1_condition), 'roi2': np.squeeze(roi2_condition)})
    
    gc_res_1to2 = grangercausalitytests(neural_ts[['roi2', 'roi1']], 1, verbose=False)
    gc_res_2to1 = grangercausalitytests(neural_ts[['roi1', 'roi2']], 1, verbose=False)
    
    f_diff = gc_res_1to2[1][0]['ssr_ftest'][0] - gc_res_2to1[1][0]['ssr_ftest'][0]
    
    return f_diff

def conduct_gca_analyses():
    for ss in subs:
        print(f"Processing subject: {ss}")
        sub_dir = f'{study_dir}/{ss}/ses-01/'
        roi_dir = f'{sub_dir}derivatives/rois'
        temp_dir = f'{raw_dir}/{ss}/ses-01/derivatives/fsl/loc'
        
        roi_coords = pd.read_csv(f'{roi_dir}/spheres/sphere_coords_hemisphere.csv')
        
        out_dir = f'{study_dir}/{ss}/ses-01/derivatives'
        os.makedirs(f'{out_dir}/gca', exist_ok=True)
        
        gca_results = []
        
        for tsk in ['loc']:
            for hemi in hemispheres:
                for rcn, rc in enumerate(run_combos):
                    pips_coords = roi_coords[(roi_coords['index'] == rcn) & 
                                             (roi_coords['task'] == tsk) & 
                                             (roi_coords['roi'] == 'pIPS') & 
                                             (roi_coords['hemisphere'] == hemi)][['x', 'y', 'z']].values.tolist()[0]
                    
                    lo_coords = roi_coords[(roi_coords['index'] == rcn) & 
                                           (roi_coords['task'] == tsk) & 
                                           (roi_coords['roi'] == 'LO') & 
                                           (roi_coords['hemisphere'] == hemi)][['x', 'y', 'z']].values.tolist()[0]
                    
                    filtered_list = [image.clean_img(nib.load(f'{temp_dir}/run-0{rn}/1stLevel.feat/filtered_func_data_reg.nii.gz'), standardize=True) for rn in rc]
                    img4d = image.concat_imgs(filtered_list)
                    
                    pips_ts = extract_roi_sphere(img4d, pips_coords)
                    lo_ts = extract_roi_sphere(img4d, lo_coords)
                    
                    object_timeseries = extract_condition_timeseries(rc, ss)
                    
                    f_diff = conduct_gca(pips_ts, lo_ts, object_timeseries)
                    gca_results.append({
                        'run_combo': rcn,
                        'hemisphere': hemi,
                        'condition': 'Object',
                        'f_diff': f_diff
                    })
        
        # Save GCA results
        gca_df = pd.DataFrame(gca_results)
        gca_df.to_csv(f'{out_dir}/gca/{ss}_gca_results.csv', index=False)
        print(f'Saved GCA results for {ss}')

if __name__ == "__main__":
    conduct_gca_analyses()