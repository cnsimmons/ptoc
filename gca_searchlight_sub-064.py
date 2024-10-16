import sys
import os
import logging
import warnings
import pandas as pd
import numpy as np
from nilearn import image, maskers
import nibabel as nib
from brainiak.searchlight.searchlight import Searchlight, Ball
from nilearn.glm.first_level import compute_regressor
from statsmodels.tsa.stattools import grangercausalitytests
import gc
import psutil
from nilearn.masking import compute_epi_mask

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Memory logging function
def log_memory_usage(step):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logging.info(f"Memory usage after {step}: {mem_info.rss / 1024 / 1024:.2f} MB")

# Set up directories and parameters
study = 'ptoc'
study_dir = f"/lab_data/behrmannlab/vlad/{study}"
localizer = 'Object'  # or 'Scramble'
results_dir = '/user_data/csimmon2/git_repos/ptoc/results'
curr_dir = f'/user_data/csimmon2/git_repos/ptoc'
sys.path.insert(0, curr_dir)
import ptoc_params as params
raw_dir = params.raw_dir

# Load subject information
sub_info = pd.read_csv(f'{curr_dir}/sub_info.csv')
sub_info = sub_info[sub_info['group'] == 'control']
#subs = sub_info['sub'].tolist()
subs = ['sub-064']  # For testing, we're using just one subject
rois = ['pIPS', 'LO']
hemispheres = ['left', 'right']

## Full run list
run_num = 3
runs = list(range(1, run_num + 1))
run_combos = [[rn1, rn2] for rn1 in range(1, run_num + 1) for rn2 in range(rn1 + 1, run_num + 1)]

# Searchlight parameters
sl_rad = 2
max_blk_edge = 10
pool_size = 1
shape = Ball

def load_and_clean_run(run_path):
    try:
        curr_run = nib.load(run_path)
        logging.info(f"Functional image: shape={curr_run.shape}, affine=\n{curr_run.affine}")
        
        mask_img = compute_epi_mask(curr_run)
        logging.info(f"Generated mask: shape={mask_img.shape}, affine=\n{mask_img.affine}")
        
        cleaned_img = image.clean_img(curr_run, standardize=True, mask_img=mask_img)
        cleaned_data = cleaned_img.get_fdata()
        
        logging.info(f"Cleaned data: shape={cleaned_data.shape}, min={np.min(cleaned_data)}, max={np.max(cleaned_data)}, mean={np.mean(cleaned_data)}")
        
        # Store the affine before deleting cleaned_img
        affine = cleaned_img.affine
        
        # Free up memory
        del curr_run, cleaned_img
        gc.collect()
        
        log_memory_usage("after loading and cleaning run")
        
        return cleaned_data, mask_img, affine
    except Exception as e:
        logging.error(f"Error in load_and_clean_run: {str(e)}")
        raise

def extract_roi_sphere(img, coords):
    roi_masker = maskers.NiftiSpheresMasker([coords], radius=6)
    seed_time_series = roi_masker.fit_transform(img)
    return seed_time_series

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
    block_ind = (cov==1).ravel()
    block_ind = np.insert(block_ind, 0, True)
    block_ind = np.delete(block_ind, len(block_ind)-1)
    block_ind = block_ind | (cov == 1).ravel()
    return ts[block_ind]

def compute_gca(data, mask, myrad, bcast_var):
    seed_ts, psy = bcast_var
    
    if isinstance(data, list):
        data = data[0]
    
    data_2d = np.array(data).reshape(-1, data.shape[-1]).T
    
    seed_phys = extract_cond_ts(seed_ts, psy)
    sphere_phys = extract_cond_ts(data_2d, psy)
    
    neural_ts = pd.DataFrame({
        'seed': seed_phys.ravel(),
        'sphere': np.mean(sphere_phys, axis=1)
    })
    
    if neural_ts['seed'].std() == 0 or neural_ts['sphere'].std() == 0:
        return np.array([np.nan])
    
    try:
        gc_res_seed = grangercausalitytests(neural_ts[['sphere', 'seed']], 1, verbose=False)
        gc_res_sphere = grangercausalitytests(neural_ts[['seed', 'sphere']], 1, verbose=False)
        f_diff = gc_res_seed[1][0]['ssr_ftest'][0] - gc_res_sphere[1][0]['ssr_ftest'][0]
        return np.array([f_diff])
    except Exception:
        return np.array([np.nan])

def conduct_searchlight():
    logging.info(f'Running searchlight analysis with GCA for {localizer}...')

    for ss in subs:
        try:
            sub_dir = f'{study_dir}/{ss}/ses-01/'
            temp_dir = f'{raw_dir}/{ss}/ses-01'
            roi_dir = f'{sub_dir}/derivatives/rois'
            exp_dir = f'{temp_dir}/derivatives/fsl/loc'
            os.makedirs(f'{sub_dir}/derivatives/gca', exist_ok=True)

            roi_coords = pd.read_csv(f'{roi_dir}/spheres/sphere_coords_hemisphere.csv')
            
            for rcn, rc in enumerate(run_combos):
                logging.info(f"Processing run combination {rc} for subject {ss}")
                
                filtered_list = []
                for rn in rc:
                    run_path = f'{exp_dir}/run-0{rn}/1stLevel.feat/filtered_func_data_reg.nii.gz'
                    logging.info(f"Loading run {rn} from {run_path}")
                    run_data, mask_img, affine = load_and_clean_run(run_path)
                    filtered_list.append(run_data)

                concat_data = np.concatenate(filtered_list, axis=-1)
                logging.info(f"Concatenated data shape: {concat_data.shape}")
                log_memory_usage("after concatenating runs")
                
                mask = mask_img.get_fdata() > 0

                psy = make_psy_cov(rc, ss)

                for roi in rois:  # rois = ['pIPS', 'LO']
                    for hemi in hemispheres:  # hemispheres = ['left', 'right']
                        coords = roi_coords[(roi_coords['index'] == rcn) & 
                                            (roi_coords['roi'] == roi) &
                                            (roi_coords['hemisphere'] == hemi)]
                        
                        if coords.empty:
                            logging.warning(f"No coordinates found for {roi}, {hemi}, run combo {rc}")
                            continue

                        coord = tuple(coords[['x', 'y', 'z']].values[0])
                        roi_ts = extract_roi_sphere(nib.Nifti1Image(concat_data, affine), coord)

                        logging.info(f"Creating Searchlight object for {roi} {hemi}...")
                        sl = Searchlight(sl_rad=sl_rad, max_blk_edge=max_blk_edge, shape=shape)
                        logging.info("Searchlight object created.")

                        logging.info("Distributing data...")
                        sl.distribute([concat_data], mask)
                        logging.info("Data distribution completed.")

                        logging.info(f"Broadcasting {roi} time series and psychological covariate...")
                        sl.broadcast((roi_ts, psy))
                        logging.info("Broadcasting completed.")

                        logging.info("Starting searchlight analysis...")
                        sl_result = sl.run_searchlight(compute_gca, pool_size=pool_size)
                        logging.info("Searchlight analysis completed.")

                        log_memory_usage("after searchlight")

                        sl_result = sl_result.astype('double')
                        sl_result[np.isnan(sl_result)] = 0
                        result_img = nib.Nifti1Image(sl_result, affine)
                        output_path = f'{sub_dir}/derivatives/gca/searchlight_result_{localizer.lower()}_runs{rc[0]}{rc[1]}_{roi}_{hemi}.nii.gz'
                        nib.save(result_img, output_path)
                        logging.info(f"Saved searchlight result to {output_path}")

                        del sl_result, roi_ts
                        gc.collect()

                del concat_data, mask
                gc.collect()

        except Exception as e:
            logging.error(f"Error processing subject {ss}: {str(e)}")
            continue

    logging.info(f'Completed searchlight analysis with GCA for all subjects')

if __name__ == "__main__":
    try:
        conduct_searchlight()
    except Exception as e:
        logging.error(f"An error occurred in the main execution: {str(e)}")