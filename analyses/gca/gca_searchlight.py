import os
import pandas as pd
import numpy as np
from nilearn import image
from nilearn.glm.first_level import compute_regressor
from statsmodels.tsa.stattools import grangercausalitytests
from scipy import stats
import sys
import nibabel as nib
import logging
from brainiak.searchlight.searchlight import Searchlight, Ball
from mpi4py import MPI
import gc
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import your parameters
curr_dir = f'/user_data/csimmon2/git_repos/ptoc'
sys.path.insert(0, curr_dir)
import ptoc_params as params

# Set up directories and parameters
study = 'ptoc'
study_dir = f"/lab_data/behrmannlab/vlad/{study}"
localizer = 'Object'  # scramble or object. This is the localizer task.
results_dir = '/user_data/csimmon2/git_repos/ptoc/results'
raw_dir = params.raw_dir

# Load subject information
sub_info = pd.read_csv(f'{curr_dir}/sub_info.csv')
sub_info = sub_info[sub_info['group'] == 'control']
#subs = sub_info['sub'].tolist()
subs = ['sub-025']

run_num = 3
runs = list(range(1, run_num + 1))
run_combos = [[rn1, rn2] for rn1 in range(1, run_num + 1) for rn2 in range(rn1 + 1, run_num + 1)]

# Searchlight parameters
sl_rad = 2
max_blk_edge = 10
pool_size = 1
shape = Ball

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
        obj_cov_file = f'{cov_dir}/catloc_{ss_num}_run-0{rn}_{localizer}.txt'

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

def gca_searchlight_kernel(data, sl_mask, myrad, bcvar):
    sl_data = data[0][sl_mask]
    sl_data = sl_data.reshape(-1, sl_data.shape[-1]).T
    
    psy = bcvar
    sl_data_cond = extract_cond_ts(sl_data, psy)
    
    n_voxels = sl_data_cond.shape[1]
    gc_matrix = np.zeros((n_voxels, n_voxels))
    
    for i in range(n_voxels):
        for j in range(i+1, n_voxels):
            voxel1 = sl_data_cond[:, i]
            voxel2 = sl_data_cond[:, j]
            
            # Check if either time series has very low variance
            if np.std(voxel1) < 1e-6 or np.std(voxel2) < 1e-6:
                continue
            
            # Check if the time series are highly correlated
            correlation, _ = stats.pearsonr(voxel1, voxel2)
            if abs(correlation) > 0.95:  # You can adjust this threshold
                continue
            
            neural_ts = pd.DataFrame({
                'voxel1': voxel1,
                'voxel2': voxel2
            })
            
            try:
                gc_res_1to2 = grangercausalitytests(neural_ts[['voxel1', 'voxel2']], 1, verbose=False)
                gc_res_2to1 = grangercausalitytests(neural_ts[['voxel2', 'voxel1']], 1, verbose=False)
                f_diff = gc_res_1to2[1][0]['ssr_ftest'][0] - gc_res_2to1[1][0]['ssr_ftest'][0]
                gc_matrix[i, j] = f_diff
                gc_matrix[j, i] = -f_diff
            except Exception as e:
                logging.warning(f"Error in GC test: {str(e)}")
                continue
    
    # If all tests failed, return 0
    if np.all(gc_matrix == 0):
        return 0
    
    return np.mean(np.abs(gc_matrix))

def conduct_gca_searchlight():
    logging.info(f'Running GCA Searchlight for {localizer}...')
    
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    for ss in subs:
        sub_dir = f'{study_dir}/{ss}/ses-01/'
        temp_dir = f'{raw_dir}/{ss}/ses-01'
        exp_dir = f'{temp_dir}/derivatives/fsl/loc'
        os.makedirs(f'{sub_dir}/derivatives/gca_searchlight', exist_ok=True)

        for rcn, rc in enumerate(run_combos):
            logging.info(f"Processing run combination {rc} for subject {ss}")
            
            filtered_list = []
            for rn in rc:
                curr_run = image.load_img(f'{exp_dir}/run-0{rn}/1stLevel.feat/filtered_func_data_reg.nii.gz')
                curr_run = image.clean_img(curr_run, standardize='zscore_sample')
                filtered_list.append(curr_run)

            img4d = image.concat_imgs(filtered_list)
            data = img4d.get_fdata()
            
            mask = (np.sum(data, axis=-1) != 0).astype(int)

            psy = make_psy_cov(rc, ss)
            
            logging.info(f"Data shape: {data.shape}")
            logging.info(f"Mask shape: {mask.shape}")
            logging.info(f"Psychological covariate shape: {psy.shape}")
            
            sl = Searchlight(sl_rad=sl_rad, max_blk_edge=max_blk_edge, shape=shape)
            sl.distribute([data], mask)
            sl.broadcast(psy)

            t1 = time.time()
            sl_result = sl.run_searchlight(gca_searchlight_kernel, pool_size=pool_size)
            logging.info(f"Searchlight duration: {(time.time()-t1)/60} minutes")

            if rank == 0:
                sl_result = sl_result.astype('double')
                sl_result[np.isnan(sl_result)] = 0
                logging.info(f"Searchlight result shape: {sl_result.shape}")
                logging.info(f"Searchlight result min: {np.min(sl_result)}, max: {np.max(sl_result)}")
                logging.info(f"Number of non-zero voxels: {np.sum(sl_result != 0)}")
                sl_img = nib.Nifti1Image(sl_result, img4d.affine)
                nib.save(sl_img, f'{sub_dir}/derivatives/gca_searchlight/gca_searchlight_result_rc{rcn+1}.nii.gz')
                logging.info(f"Saved searchlight results for subject {ss}, run combo {rcn+1}")

            gc.collect()

        logging.info(f'Completed GCA Searchlight for subject {ss}')

def summarize_gca_searchlight():
    logging.info('Creating summary across subjects...')
    
    all_subjects_data = []
    
    for ss in subs:
        sub_dir = f'{study_dir}/{ss}/ses-01/'
        data_dir = f'{sub_dir}/derivatives/gca_searchlight'
        
        for rcn in range(len(run_combos)):
            result_file = f'{data_dir}/gca_searchlight_result_rc{rcn+1}.nii.gz'
            if os.path.exists(result_file):
                img = nib.load(result_file)
                data = img.get_fdata()
                all_subjects_data.append(data)
    
    if all_subjects_data:
        mean_data = np.mean(all_subjects_data, axis=0)
        std_data = np.std(all_subjects_data, axis=0)
        
        output_dir = f"{results_dir}/gca_searchlight"
        os.makedirs(output_dir, exist_ok=True)
        
        mean_img = nib.Nifti1Image(mean_data, img.affine)
        std_img = nib.Nifti1Image(std_data, img.affine)
        
        nib.save(mean_img, f"{output_dir}/all_subjects_gca_searchlight_mean_{localizer.lower()}.nii.gz")
        nib.save(std_img, f"{output_dir}/all_subjects_gca_searchlight_std_{localizer.lower()}.nii.gz")
        
        logging.info(f'Summary across subjects completed and saved to {output_dir}')
    else:
        logging.error("No valid data found for any subjects. Cannot create summary.")

if __name__ == "__main__":
    conduct_gca_searchlight()
    #summarize_gca_searchlight()