import os
import pandas as pd
import numpy as np
from nilearn import image, input_data
from nilearn.glm.first_level import compute_regressor
from statsmodels.tsa.stattools import grangercausalitytests
import sys
import nibabel as nib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import your parameters
curr_dir = f'/user_data/csimmon2/git_repos/ptoc'
sys.path.insert(0, curr_dir)
import ptoc_params as params

# Set up directories and parameters
study = 'ptoc'
study_dir = f"/lab_data/behrmannlab/vlad/{study}"
results_dir = '/user_data/csimmon2/git_repos/ptoc/results'
raw_dir = params.raw_dir

# Load subject information
sub_info = pd.read_csv(f'{curr_dir}/sub_info.csv')
sub_info = sub_info[sub_info['group'] == 'control']
#subs = sub_info['sub'].tolist()
subs = ['sub-025', 'sub-038']

rois = ['pIPS', 'LO']
hemispheres = ['left', 'right']
run_num = 3
runs = list(range(1, run_num + 1))
run_combos = [[rn1, rn2] for rn1 in range(1, run_num + 1) for rn2 in range(rn1 + 1, run_num + 1)]

def extract_roi_sphere(img, coords):
    roi_masker = input_data.NiftiSpheresMasker([tuple(coords)], radius=6)
    seed_time_series = roi_masker.fit_transform(img)
    phys = np.mean(seed_time_series, axis=1).reshape(-1, 1)
    return phys  # Return non-standardized time series

def make_psy_cov(runs, ss):
    temp_dir = f'{raw_dir}/{ss}/ses-01'
    cov_dir = f'{temp_dir}/covs'
    vols_per_run, tr = 184, 2.0
    total_vols = vols_per_run * len(runs)
    times = np.arange(0, total_vols * tr, tr)
    full_cov = pd.DataFrame(columns=['onset', 'duration', 'value'])

    for i, rn in enumerate(runs):
        ss_num = ss.split('-')[1]
        obj_cov_file = f'{cov_dir}/catloc_{ss_num}_run-0{rn}_Object.txt'

        if not os.path.exists(obj_cov_file):
            logging.warning(f'Covariate file not found for run {rn}')
            continue

        obj_cov = pd.read_csv(obj_cov_file, sep='\t', header=None, names=['onset', 'duration', 'value'])
        
        if i > 0:
            obj_cov['onset'] += i * vols_per_run * tr
        
        full_cov = pd.concat([full_cov, obj_cov])

    full_cov = full_cov.sort_values(by=['onset']).reset_index(drop=True)
    cov = full_cov.to_numpy()
    valid_onsets = cov[:, 0] < times[-1]
    cov = cov[valid_onsets]

    if cov.shape[0] == 0:
        logging.warning('No valid covariate data after filtering. Returning zeros array.')
        return np.zeros((total_vols, 1))

    psy, _ = compute_regressor(cov.T, 'spm', times)
    psy[psy > 0] = 1
    psy[psy <= 0] = 0
    return psy

def extract_cond_ts(ts, cov):
    block_ind = (cov==1)
    block_ind = np.insert(block_ind, 0, True)
    block_ind = np.delete(block_ind, len(block_ind)-1)
    block_ind = (cov == 1).reshape((len(cov))) | block_ind
    return ts[block_ind]

def conduct_gca():
    logging.info('Running GCA...')
    tasks = ['loc']
    
    for ss in subs:
        sub_summary = pd.DataFrame(columns=['sub', 'fold', 'task', 'origin', 'target', 'f_diff'])
        
        sub_dir = f'{study_dir}/{ss}/ses-01/'
        temp_dir = f'{raw_dir}/{ss}/ses-01'
        roi_dir = f'{sub_dir}/derivatives/rois'
        exp_dir = f'{temp_dir}/derivatives/fsl/loc'
        os.makedirs(f'{sub_dir}/derivatives/results/gca', exist_ok=True)

        roi_coords = pd.read_csv(f'{roi_dir}/spheres/sphere_coords_hemisphere.csv')
        logging.info(f"ROI coordinates loaded for subject {ss}")

        for rcn, rc in enumerate(run_combos):
            logging.info(f"Processing run combination {rc} for subject {ss}")
            
            filtered_list = []
            for rn in rc:
                curr_run = image.load_img(f'{exp_dir}/run-0{rn}/1stLevel.feat/filtered_func_data_reg.nii.gz')
                curr_run = image.clean_img(curr_run, standardize=True)
                filtered_list.append(curr_run)

            img4d = image.concat_imgs(filtered_list)
            logging.info(f"Concatenated image shape: {img4d.shape}")

            for tsk in tasks:
                for dorsal_roi in ['pIPS']:
                    for dorsal_hemi in hemispheres:
                        dorsal_coords = roi_coords[(roi_coords['index'] == rcn) & 
                                                   (roi_coords['task'] == tsk) & 
                                                   (roi_coords['roi'] == dorsal_roi) &
                                                   (roi_coords['hemisphere'] == dorsal_hemi)]
                        
                        if dorsal_coords.empty:
                            logging.warning(f"No coordinates found for {dorsal_roi}, {dorsal_hemi}, run combo {rc}")
                            continue

                        dorsal_ts = extract_roi_sphere(img4d, dorsal_coords[['x', 'y', 'z']].values.tolist()[0])
                        
                        psy = make_psy_cov(rc, ss)
                        
                        if dorsal_ts.shape[0] != psy.shape[0]:
                            raise ValueError(f"Mismatch in volumes: dorsal_ts has {dorsal_ts.shape[0]}, psy has {psy.shape[0]}")
                        
                        dorsal_phys = extract_cond_ts(dorsal_ts, psy)
                        
                        for ventral_roi in ['LO']:
                            for ventral_hemi in hemispheres:
                                ventral_coords = roi_coords[(roi_coords['index'] == rcn) & 
                                                            (roi_coords['task'] == tsk) & 
                                                            (roi_coords['roi'] == ventral_roi) &
                                                            (roi_coords['hemisphere'] == ventral_hemi)]
                                
                                if ventral_coords.empty:
                                    logging.warning(f"No coordinates found for {ventral_roi}, {ventral_hemi}, run combo {rc}")
                                    continue
                                
                                ventral_ts = extract_roi_sphere(img4d, ventral_coords[['x', 'y', 'z']].values.tolist()[0])
                                ventral_phys = extract_cond_ts(ventral_ts, psy)

                                neural_ts = pd.DataFrame({
                                    'dorsal': dorsal_phys.ravel(), 
                                    'ventral': ventral_phys.ravel()
                                })
                                
                                gc_res_dorsal = grangercausalitytests(neural_ts[['ventral', 'dorsal']], 1, verbose=False)
                                gc_res_ventral = grangercausalitytests(neural_ts[['dorsal', 'ventral']], 1, verbose=False)

                                f_diff = gc_res_dorsal[1][0]['ssr_ftest'][0] - gc_res_ventral[1][0]['ssr_ftest'][0]

                                if abs(f_diff) > 10:  # Adjust this threshold as needed
                                    logging.warning(f"Large F-diff value ({f_diff}) detected for {ss}, {tsk}, {dorsal_roi}_{dorsal_hemi}, {ventral_roi}_{ventral_hemi}")

                                dorsal_label = f"{dorsal_hemi[0]}{dorsal_roi}"
                                ventral_label = f"{ventral_hemi[0]}{ventral_roi}"
                                curr_data = pd.Series([ss, rcn, tsk, dorsal_label, ventral_label, f_diff], index=sub_summary.columns)
                                
                                sub_summary = sub_summary.append(curr_data, ignore_index=True)
                                logging.info(f"Completed GCA for {ss}, {tsk}, {dorsal_label}, {ventral_label}")

        logging.info(f'Completed GCA for subject {ss}')
        sub_summary.to_csv(f'{sub_dir}/derivatives/results/gca/gca_summary_trial.csv', index=False)

def summarize_gca():
    logging.info('Creating summary across subjects...')
    
    all_subjects_data = []
    
    for ss in subs:
        sub_dir = f'{study_dir}/{ss}/ses-01/'
        data_dir = f'{sub_dir}/derivatives/results/gca'
        
        curr_df = pd.read_csv(f'{data_dir}/gca_summary_trial.csv')
        curr_df['sub'] = ss
        all_subjects_data.append(curr_df)
    
    df_all = pd.concat(all_subjects_data, ignore_index=True)
    
    df_summary = df_all.groupby(['fold', 'task', 'origin', 'target'])['f_diff'].agg(['mean', 'std']).reset_index()
    
    df_summary.columns = ['fold', 'task', 'origin', 'target', 'mean_f_diff', 'std_f_diff']
    df_summary = df_summary.sort_values(['fold', 'task', 'origin', 'target'])
    
    output_dir = f"{results_dir}/gca"
    os.makedirs(output_dir, exist_ok=True)
    summary_file = f"{output_dir}/all_subjects_gca_summary_trial.csv"
    df_summary.to_csv(summary_file, index=False)
    
    logging.info(f'Summary across subjects completed and saved to {summary_file}')
    print(df_summary)
    
    return df_summary

# Main execution
if __name__ == "__main__":
    conduct_gca()
    summarize_gca()