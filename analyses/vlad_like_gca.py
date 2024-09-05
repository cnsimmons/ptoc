import os
import pandas as pd
import numpy as np
from nilearn import image, input_data
from nilearn.glm.first_level import compute_regressor
from statsmodels.tsa.stattools import grangercausalitytests
import sys
import nibabel as nib
import itertools

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
subs = ['sub-025']  # Updated list of subjects
rois = ['pIPS', 'LO']
hemispheres = ['left', 'right']
run_num = 3
runs = list(range(1, run_num + 1))
run_combos = [[rn1, rn2] for rn1 in range(1, run_num + 1) for rn2 in range(rn1 + 1, run_num + 1)]

def extract_roi_sphere(img, coords):
    roi_masker = input_data.NiftiSpheresMasker([tuple(coords)], radius=6)
    seed_time_series = roi_masker.fit_transform(img)
    phys = np.mean(seed_time_series, axis=1).reshape(-1, 1)
    return phys

def make_psy_cov(runs, ss):
    temp_dir = f'{raw_dir}/{ss}/ses-01'
    cov_dir = f'{temp_dir}/covs'
    vols, tr = 184, 2.0
    times = np.arange(0, vols * tr, tr)
    full_cov = pd.DataFrame(columns=['onset', 'duration', 'value'])

    for rn in runs:
        ss_num = ss.split('-')[1]
        obj_cov_file = f'{cov_dir}/catloc_{ss_num}_run-0{rn}_Object.txt'
        scr_cov_file = f'{cov_dir}/catloc_{ss_num}_run-0{rn}_Scramble.txt'

        if not os.path.exists(obj_cov_file) or not os.path.exists(scr_cov_file):
            print(f'Covariate file not found for run {rn}')
            continue

        obj_cov = pd.read_csv(obj_cov_file, sep='\t', header=None, names=['onset', 'duration', 'value'])
        scr_cov = pd.read_csv(scr_cov_file, sep='\t', header=None, names=['onset', 'duration', 'value'])
        scr_cov['value'] *= -1
        full_cov = pd.concat([full_cov, obj_cov, scr_cov])

    full_cov = full_cov.sort_values(by=['onset']).reset_index(drop=True)
    cov = full_cov.to_numpy()
    valid_onsets = cov[:, 0] < times[-1]
    cov = cov[valid_onsets]

    if cov.shape[0] == 0:
        print('No valid covariate data after filtering. Returning zeros array.')
        return np.zeros((vols, 1))

    psy, _ = compute_regressor(cov.T, 'spm', times)
    return psy

def extract_cond_ts(ts, cov):
    """
    Extracts timeseries corresponding to blocks in a cov file
    """
    block_ind = (cov == 1)
    block_ind = np.insert(block_ind, 0, True)
    block_ind = np.delete(block_ind, len(block_ind)-1)
    block_ind = (cov == 1).reshape((len(cov))) | block_ind

    new_ts = ts[block_ind]
    return new_ts

def conduct_gca():
    print('Running GCA...')
    tasks = ['loc']
    cond = ['Object']
    
    for ss in subs:
        sub_summary = pd.DataFrame(columns=['sub', 'fold', 'task', 'origin', 'target', 'f_diff'])
        
        sub_dir = f'{study_dir}/{ss}/ses-01/'
        temp_dir = f'{raw_dir}/{ss}/ses-01'
        cov_dir = f'{temp_dir}/covs'
        roi_dir = f'{sub_dir}/derivatives/rois'
        exp_dir = f'{sub_dir}/derivatives/fsl/loc'
        os.makedirs(f'{sub_dir}/derivatives/results/beta_ts', exist_ok=True)

        roi_coords = pd.read_csv(f'{roi_dir}/spheres/sphere_coords_hemisphere.csv')
        #print("ROI coordinates DataFrame:")
        #print(roi_coords)
        #print("\nROI coordinates shape:", roi_coords.shape)
        #print("\nROI coordinates columns:", roi_coords.columns)

        for rcn, rc in enumerate(run_combos):
            # Extract timeseries from each run
            filtered_list = []
            for rn in rc:
                curr_run = image.load_img(f'{exp_dir}/run-0{rn}/1stLevel.feat/filtered_func_data_reg.nii.gz')
                curr_run = image.clean_img(curr_run, standardize=True)
                filtered_list.append(curr_run)

            # concat runs
            img4d = image.concat_imgs(filtered_list)

            print(ss, rcn, 'loaded')

            for tsk in tasks:
                for dorsal_roi in ['pIPS']:
                    for dorsal_hemi in hemispheres:
                        # load peak voxel in dorsal roi
                        dorsal_coords = roi_coords[(roi_coords['index'] == rcn) & 
                                                   (roi_coords['task'] == tsk) & 
                                                   (roi_coords['roi'] == dorsal_roi) &
                                                   (roi_coords['hemisphere'] == dorsal_hemi)]
                        
                        print(f"\nFiltered dorsal coordinates for rcn={rcn}, task={tsk}, roi={dorsal_roi}, hemisphere={dorsal_hemi}:")
                        print(dorsal_coords)
                        print("Filtered dorsal coordinates shape:", dorsal_coords.shape)

                        if dorsal_coords.empty:
                            print(f"Warning: No coordinates found for rcn={rcn}, task={tsk}, roi={dorsal_roi}, hemisphere={dorsal_hemi}")
                            continue

                        # Extract TS from dorsal roi
                        dorsal_ts = extract_roi_sphere(img4d, dorsal_coords[['x', 'y', 'z']].values.tolist()[0])

                        
                        # load behavioral data
                        # time adjusted using HRF to pull out boxcar
                        psy = make_psy_cov(rc, ss)

                        # create dorsal ts for just that condition
                        dorsal_phys = extract_cond_ts(dorsal_ts, psy)
                        
                        for ventral_roi in ['LO']:
                            for ventral_hemi in hemispheres:
                                ventral_coords = roi_coords[(roi_coords['index'] == rcn) & 
                                                            (roi_coords['task'] == 'loc') & 
                                                            (roi_coords['roi'] == ventral_roi) &
                                                            (roi_coords['hemisphere'] == ventral_hemi)]
                                
                                if ventral_coords.empty:
                                    print(f"Warning: No coordinates found for rcn={rcn}, task={tsk}, roi={ventral_roi}, hemisphere={ventral_hemi}")
                                    continue
                                
                                ventral_ts = extract_roi_sphere(img4d, ventral_coords[['x', 'y', 'z']].values.tolist()[0])
                                ventral_phys = extract_cond_ts(ventral_ts, psy)                            

                                # Add TSs to a dataframe to prep for gca
                                neural_ts = pd.DataFrame(columns=['dorsal', 'ventral'])
                                neural_ts['dorsal'] = np.squeeze(dorsal_phys)
                                neural_ts['ventral'] = np.squeeze(ventral_phys)
                                
                                # calculate dorsal GCA F-test
                                gc_res_dorsal = grangercausalitytests(neural_ts[['ventral', 'dorsal']], 1, verbose=False)
                                
                                # calculate ventral GCA F-test
                                gc_res_ventral = grangercausalitytests(neural_ts[['dorsal', 'ventral']], 1, verbose=False)

                                # calc difference
                                f_diff = gc_res_dorsal[1][0]['ssr_ftest'][0] - gc_res_ventral[1][0]['ssr_ftest'][0]

                                dorsal_label = f"{dorsal_hemi[0]}{dorsal_roi}"
                                ventral_label = f"{ventral_hemi[0]}{ventral_roi}"
                                curr_data = pd.Series([ss, rcn, tsk, dorsal_label, ventral_label, f_diff], index=sub_summary.columns)
                                
                                sub_summary = sub_summary.append(curr_data, ignore_index=True)
                                print(ss, tsk, dorsal_label, ventral_label)

        print('done GCA for', ss)                
        sub_summary.to_csv(f'{sub_dir}/derivatives/results/beta_ts/gca_summary.csv', index=False)

def summarize_gca():
    """
    Compile subject data into one summary
    """
    print('Creating summary across subjects...')
    
    df_summary = pd.DataFrame()
    tasks = ['loc']
    cond = ['Object']
    
    print(subs)
    for ss in subs:
        sub_dir = f'{study_dir}/{ss}/ses-01/'
        data_dir = f'{sub_dir}/derivatives/results/beta_ts'

        curr_df = pd.read_csv(f'{data_dir}/gca_summary.csv')
        curr_df = curr_df.groupby(['task','origin','target']).mean()
        curr_data = [ss]
        col_index = ['sub']
        for tsk in tasks:
            for dorsal_roi in ['pIPS']:
                for dorsal_hemi in hemispheres:
                    for ventral_roi in ['LO']:
                        for ventral_hemi in hemispheres:
                            dorsal_label = f"{dorsal_hemi[0]}{dorsal_roi}"
                            ventral_label = f"{ventral_hemi[0]}{ventral_roi}"
                            col_index.append(f'{tsk}_{dorsal_label}_{ventral_label}')
                            curr_data.append(curr_df['f_diff'][tsk, dorsal_label, ventral_label])

        if ss == subs[0]:
            df_summary = pd.DataFrame(columns=col_index)
            df_summary = df_summary.append(pd.Series(curr_data,index = col_index),ignore_index=True)
        else:
            df_summary = df_summary.append(pd.Series(curr_data,index = col_index),ignore_index=True)

    df_summary.to_csv(f"{results_dir}/gca/all_roi_summary.csv", index=False)

# Main execution
subs = ['sub-025']
conduct_gca()
summarize_gca()