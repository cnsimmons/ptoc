import warnings
warnings.filterwarnings("ignore")
import resource
import sys
import time
import os
import gc
import pandas as pd
import numpy as np
import logging
from nilearn import image, input_data
from nilearn.glm.first_level import compute_regressor
from statsmodels.tsa.stattools import grangercausalitytests
import nibabel as nib
from brainiak.searchlight.searchlight import Searchlight, Ball

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import parameters
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
subs = ['sub-025']  # You can expand this list if needed

# Other parameters
rois = ['pIPS', 'LO']
hemispheres = ['left', 'right']
run_num = 3
runs = list(range(1, run_num + 1))
run_combos = [[rn1, rn2] for rn1 in range(1, run_num + 1) for rn2 in range(rn1 + 1, run_num + 1)]

whole_brain_mask = image.load_img('/user_data/csimmon2/git_repos/ptoc/roiParcels/mruczek_parcels/binary/all_visual_areas.nii.gz')
affine = whole_brain_mask.affine
dimsizes = whole_brain_mask.header.get_zooms()  # get dimensions

# scan parameters
vols = 184
tr = 2.0

print("Setting up searchlight...")
mask = image.get_data(whole_brain_mask)  # the mask to search within
sl_rad = 2  # searchlight radius in voxels
max_blk_edge = 10  # how many blocks to send on each parallelized search
pool_size = 1  # how many cores to use

voxels_proportion = 1
shape = Ball

def extract_roi_sphere(img, coords):
    roi_masker = input_data.NiftiSpheresMasker([tuple(coords)], radius=6)
    seed_time_series = roi_masker.fit_transform(img)
    phys = np.mean(seed_time_series, axis=1).reshape(-1, 1)
    phys_standardized = phys
    return phys_standardized

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
    psy[psy < 0] = 0  
    
    return psy

def extract_cond_ts(ts, cov):
    block_ind = (cov==1)
    block_ind = np.insert(block_ind, 0, True)
    block_ind = np.delete(block_ind, len(block_ind)-1)
    block_ind = (cov == 1).reshape((len(cov))) | block_ind
    return ts[block_ind]

def gca(data, sl_mask, myrad, seed_ts, psy):
    # This function will be called for each searchlight sphere
    sphere_ts = data[0]
    
    # Extract condition-specific time series for the sphere and seed ROIs
    sphere_phys = extract_cond_ts(sphere_ts, psy)
    dorsal_phys = extract_cond_ts(seed_ts['dorsal'], psy)
    ventral_phys = extract_cond_ts(seed_ts['ventral'], psy)
    
    # Compute GCA between sphere and seed ROIs
    neural_ts_dorsal = pd.DataFrame({'sphere': sphere_phys.ravel(), 'dorsal': dorsal_phys.ravel()})
    neural_ts_ventral = pd.DataFrame({'sphere': sphere_phys.ravel(), 'ventral': ventral_phys.ravel()})
    
    gc_res_dorsal = grangercausalitytests(neural_ts_dorsal, 1, verbose=False)
    gc_res_ventral = grangercausalitytests(neural_ts_ventral, 1, verbose=False)
    
    f_diff_dorsal = gc_res_dorsal[1][0]['ssr_ftest'][0]
    f_diff_ventral = gc_res_ventral[1][0]['ssr_ftest'][0]
    
    return np.array([f_diff_dorsal, f_diff_ventral])

def process_subject(ss):
    logging.info(f'Processing subject {ss}')
    sub_dir = f'{study_dir}/{ss}/ses-01/'
    temp_dir = f'{raw_dir}/{ss}/ses-01'
    roi_dir = f'{sub_dir}/derivatives/rois'
    exp_dir = f'{temp_dir}/derivatives/fsl/loc'
    os.makedirs(f'{sub_dir}/derivatives/gca_searchlight', exist_ok=True)

    roi_coords = pd.read_csv(f'{roi_dir}/spheres/sphere_coords_hemisphere.csv')

    for rcn, rc in enumerate(run_combos):
        logging.info(f"Processing run combination {rc} for subject {ss}")
        
        # Load and preprocess data
        filtered_list = []
        for rn in rc:
            curr_run = image.load_img(f'{exp_dir}/run-0{rn}/1stLevel.feat/filtered_func_data_reg.nii.gz')
            curr_run = image.clean_img(curr_run, standardize=True)
            filtered_list.append(curr_run)

        img4d = image.concat_imgs(filtered_list)
        bold_vol = img4d.get_fdata()
        
        # Extract ROI time series
        dorsal_ts = extract_roi_sphere(img4d, roi_coords[(roi_coords['roi'] == 'pIPS') & (roi_coords['hemisphere'] == 'left')][['x', 'y', 'z']].values.tolist()[0])
        ventral_ts = extract_roi_sphere(img4d, roi_coords[(roi_coords['roi'] == 'LO') & (roi_coords['hemisphere'] == 'left')][['x', 'y', 'z']].values.tolist()[0])
        
        psy = make_psy_cov(rc, ss)
        
        seed_ts = {'dorsal': dorsal_ts, 'ventral': ventral_ts}
        
        # Set up and run searchlight
        sl = Searchlight(sl_rad=sl_rad, max_blk_edge=max_blk_edge, shape=shape)
        sl.distribute([bold_vol], mask)
        sl.broadcast(seed_ts)
        sl.broadcast(psy)  # Broadcast psy to all processes
        
        # Modify the gca function call to include psy
        sl_result = sl.run_searchlight(gca, pool_size=pool_size)
        
        # Save results
        result_img = nib.Nifti1Image(sl_result, affine)
        nib.save(result_img, f'{sub_dir}/derivatives/gca_searchlight/gca_searchlight_result_rc{rcn}.nii.gz')

    logging.info(f'Completed processing for subject {ss}')

# Main execution
if __name__ == "__main__":
    for ss in subs:
        process_subject(ss)