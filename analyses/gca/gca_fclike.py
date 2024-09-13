import os
import pandas as pd
import numpy as np
from nilearn import image, input_data
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

##TO RUN ALL
sub_info = pd.read_csv(f'{curr_dir}/sub_info.csv')
sub_info = sub_info[sub_info['group'] == 'control']
#subs = sub_info['sub'].tolist()
#subs = [sub for sub in all_subs if sub != 'sub-025'] #all subs but 25

#TO RUN ONE
subs = ['sub-025']

rois = ['pIPS', 'LO']
hemispheres = ['left', 'right']
run_num = 3
runs = list(range(1, run_num + 1))
run_combos = [[rn1, rn2] for rn1 in range(1, run_num + 1) for rn2 in range(rn1 + 1, run_num + 1)]

def standardize_ts(ts):
    """
    Standardize timeseries to have zero mean and unit variance.
    """
    return (ts - np.mean(ts)) / np.std(ts)

def check_variance(ts, label):
    variance = np.var(ts)
    if variance < 0.9 or variance > 1.1:  # allowing for some numerical imprecision
        logging.warning(f"Unusual variance ({variance}) detected for {label} timeseries")

def extract_roi_sphere(img, coords):
    roi_masker = input_data.NiftiSpheresMasker([tuple(coords)], radius=6)
    seed_time_series = roi_masker.fit_transform(img)
    phys = np.mean(seed_time_series, axis=1).reshape(-1, 1)
    phys_standardized = standardize_ts(phys)
    return phys_standardized

def conduct_fc_gca():
    logging.info('Running FC-GCA...')
    tasks = ['loc']
    
    for ss in subs:
        sub_summary = pd.DataFrame(columns=['sub', 'fold', 'task', 'origin', 'target', 'f_diff'])
        
        sub_dir = f'{study_dir}/{ss}/ses-01/'
        temp_dir = f'{raw_dir}/{ss}/ses-01'
        roi_dir = f'{sub_dir}/derivatives/rois'
        exp_dir = f'{temp_dir}/derivatives/fsl/loc'
        os.makedirs(f'{sub_dir}/derivatives/results/fc_gca', exist_ok=True)

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
                        check_variance(dorsal_ts, f"{ss}_{tsk}_{dorsal_roi}_{dorsal_hemi}")
                        
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
                                check_variance(ventral_ts, f"{ss}_{tsk}_{ventral_roi}_{ventral_hemi}")

                                neural_ts = pd.DataFrame({
                                    'dorsal': dorsal_ts.ravel(), 
                                    'ventral': ventral_ts.ravel()
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
                                logging.info(f"Completed FC-GCA for {ss}, {tsk}, {dorsal_label}, {ventral_label}")

        logging.info(f'Completed FC-GCA for subject {ss}')
        sub_summary.to_csv(f'{sub_dir}/derivatives/results/fc_gca/gca_summary_fc.csv', index=False)

def summarize_fc_gca():
    logging.info('Creating summary across subjects...')
    
    all_subjects_data = []
    
    for ss in subs:
        sub_dir = f'{study_dir}/{ss}/ses-01/'
        data_dir = f'{sub_dir}/derivatives/results/fc_gca'
        
        curr_df = pd.read_csv(f'{data_dir}/gca_summary_fc.csv')
        curr_df['sub'] = ss
        all_subjects_data.append(curr_df)
    
    df_all = pd.concat(all_subjects_data, ignore_index=True)
    
    df_summary = df_all.groupby(['fold', 'task', 'origin', 'target'])['f_diff'].agg(['mean', 'std']).reset_index()
    
    df_summary.columns = ['fold', 'task', 'origin', 'target', 'mean_f_diff', 'std_f_diff']
    df_summary = df_summary.sort_values(['fold', 'task', 'origin', 'target'])
    
    output_dir = f"{results_dir}/fc_gca"
    os.makedirs(output_dir, exist_ok=True)
    summary_file = f"{output_dir}/all_subjects_gca_summary_fc.csv"
    df_summary.to_csv(summary_file, index=False)
    
    logging.info(f'Summary across subjects completed and saved to {summary_file}')
    print(df_summary)
    
    return df_summary

# Main execution
if __name__ == "__main__":
    conduct_fc_gca()
    #summarize_fc_gca()