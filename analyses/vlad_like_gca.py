import os
import pandas as pd
import numpy as np
from nilearn import image, input_data, glm
from statsmodels.tsa.stattools import grangercausalitytests
import sys
import nibabel as nib
import warnings

warnings.filterwarnings('ignore')

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
rois = ['pIPS', 'LO']
hemispheres = ['left', 'right']
run_num = 3
runs = list(range(1, run_num + 1))
run_combos = [[rn1, rn2] for rn1 in range(1, run_num + 1) for rn2 in range(rn1 + 1, run_num + 1)]

def extract_roi_sphere(img, coords):
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
        
        obj_cov['onset'] += (rn - 1) * vols * tr
        
        cov = obj_cov.to_numpy().astype(float)
        psy, _ = glm.first_level.compute_regressor(cov.T, 'spm', times)
        psy[psy > 0] = 1
        psy[psy <= 0] = 0
        
        object_timeseries += psy.flatten()
    
    return object_timeseries

def extract_cond_ts(ts, cov):
    block_ind = (cov == 1)
    block_ind = np.insert(block_ind, 0, True)
    block_ind = np.delete(block_ind, len(block_ind) - 1)
    block_ind = (cov == 1).reshape((len(cov))) | block_ind
    return ts[block_ind]

def conduct_gca():
    print('Running GCA...')
    
    for ss in subs:
        sub_summary = pd.DataFrame(columns=['sub', 'fold', 'condition', 'origin', 'target', 'f_diff'])
        
        sub_dir = f'{study_dir}/{ss}/ses-01/'
        roi_dir = f'{sub_dir}derivatives/rois'
        temp_dir = f'{raw_dir}/{ss}/ses-01/derivatives/fsl/loc'
        
        out_dir = f'{study_dir}/{ss}/ses-01/derivatives'
        os.makedirs(f'{out_dir}/gca_standard', exist_ok=True)
        
        roi_coords_path = f'{study_dir}/{ss}/ses-01/derivatives/rois/spheres/sphere_coords_std.csv'
        if not os.path.exists(roi_coords_path):
            print(f"ROI coordinates file not found: {roi_coords_path}")
            continue
        
        roi_coords = pd.read_csv(roi_coords_path)
        
        for rcn, rc in enumerate(run_combos):
            filtered_list = [image.clean_img(nib.load(f'{temp_dir}/run-0{rn}/1stLevel.feat/filtered_func_data_reg.nii.gz'), standardize=True) for rn in rc]
            img4d = image.concat_imgs(filtered_list)
            
            object_timeseries = extract_condition_timeseries(rc, ss)
            
            for hemi in hemispheres:
                pips_coords = roi_coords[(roi_coords['roi'] == 'pIPS') & (roi_coords['hemisphere'] == hemi)][['x', 'y', 'z']].values[0]
                lo_coords = roi_coords[(roi_coords['roi'] == 'LO') & (roi_coords['hemisphere'] == hemi)][['x', 'y', 'z']].values[0]
                
                pips_ts = extract_roi_sphere(img4d, pips_coords)
                lo_ts = extract_roi_sphere(img4d, lo_coords)
                
                pips_cond = extract_cond_ts(pips_ts, object_timeseries)
                lo_cond = extract_cond_ts(lo_ts, object_timeseries)
                
                neural_ts = pd.DataFrame({'pIPS': np.squeeze(pips_cond), 'LO': np.squeeze(lo_cond)})
                
                gc_res_pips_to_lo = grangercausalitytests(neural_ts[['LO', 'pIPS']], 1, verbose=False)
                gc_res_lo_to_pips = grangercausalitytests(neural_ts[['pIPS', 'LO']], 1, verbose=False)
                
                f_diff = gc_res_pips_to_lo[1][0]['ssr_ftest'][0] - gc_res_lo_to_pips[1][0]['ssr_ftest'][0]
                
                curr_data = pd.Series([ss, rcn, 'Object', 'pIPS', 'LO', f_diff], index=sub_summary.columns)
                sub_summary = sub_summary.append(curr_data, ignore_index=True)
                
                print(f"Processed {ss}, {hemi} hemisphere, run combo {rcn}")
        
        print(f'Done GCA for {ss}')
        sub_summary.to_csv(f'{out_dir}/gca_standard/{ss}_gca_results_standard.csv', index=False)

def summarize_gca():
    print('Creating summary across subjects...')
    
    df_summary = pd.DataFrame(columns=['sub'] + [f'{hemi}_{roi1}_{roi2}' for hemi in hemispheres for roi1 in rois for roi2 in rois if roi1 != roi2])
    
    for ss in subs:
        sub_dir = f'{study_dir}/{ss}/ses-01/'
        data_dir = f'{sub_dir}/derivatives/gca_standard'
        
        curr_df = pd.read_csv(f'{data_dir}/{ss}_gca_results_standard.csv')
        curr_df = curr_df.groupby(['condition', 'origin', 'target']).mean()
        
        curr_data = [ss]
        for hemi in hemispheres:
            for roi1 in rois:
                for roi2 in rois:
                    if roi1 != roi2:
                        col_name = f'{hemi}_{roi1}_{roi2}'
                        curr_data.append(curr_df['f_diff']['Object', roi1, roi2])
        
        df_summary = df_summary.append(pd.Series(curr_data, index=df_summary.columns), ignore_index=True)
    
    df_summary.to_csv(f"{results_dir}/gca/all_roi_summary_standard.csv", index=False)

if __name__ == "__main__":
    conduct_gca()
    summarize_gca()
    print("GCA analysis completed.")