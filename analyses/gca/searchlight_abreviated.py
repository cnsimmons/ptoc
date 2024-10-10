import warnings
warnings.filterwarnings("ignore")
import resource
import sys
import time
import os
import gc
import pandas as pd
import numpy as np
import nibabel as nib
from nilearn import image
from brainiak.searchlight.searchlight import Searchlight, Ball
from statsmodels.tsa.stattools import grangercausalitytests

# Import your parameters
curr_dir = f'/user_data/csimmon2/git_repos/ptoc'
sys.path.insert(0, curr_dir)
import ptoc_params as params

print('Libraries loaded...')

# Load subject information
sub_info = pd.read_csv(f'{curr_dir}/sub_info.csv')
sub_info = sub_info[sub_info['group'] == 'control']
subs = ['sub-025']  # Run for one subject
dorsal = ['pIPS']  # Run for one ROI

print(subs, dorsal)

# Set up directories and parameters
study = 'ptoc'
study_dir = f"/lab_data/behrmannlab/vlad/{study}"
raw_dir = "/lab_data/behrmannlab/vlad/hemispace"
exp = 'loc'

out_dir = f'{study_dir}/derivatives/fc'
sub_dir = f'{study_dir}/sub-025/ses-01/'
exp_dir = f'{sub_dir}/derivatives/fsl/{exp}'

runs = list(range(1,3))

# Add this near the top of your script, after imports
gca_counter = 0

def transform_mask_to_native(subject_func, standard_mask, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    func_img = image.load_img(subject_func)
    mask_img = image.load_img(standard_mask)
    native_mask = image.resample_to_img(mask_img, func_img, interpolation='nearest')
    output_path = os.path.join(output_dir, f'whole_brain_mask_native.nii.gz')
    native_mask.to_filename(output_path)
    return output_path

def gca_searchlight(data, sl_mask, myrad, seed_ts):
    global gca_counter
    gca_counter += 1
    
    print(f"GCA analysis {gca_counter}: Starting...")
    target_data = data[0]
    target_voxels = np.sum(sl_mask != 0)
    print(f"Voxels in target sphere: {target_voxels}")
    
    target_ts = target_data.reshape(-1, target_data.shape[-1]).T
    target_mean_ts = np.mean(target_ts, axis=1)
    
    try:
        neural_ts = pd.DataFrame({'seed': seed_ts, 'target': target_mean_ts})
        
        for col in neural_ts.columns:
            if neural_ts[col].std() == 0:
                neural_ts[col] += np.random.normal(0, 1e-10, len(neural_ts[col]))
        
        gc_res_seed_to_target = grangercausalitytests(neural_ts[['seed', 'target']], 1, verbose=False)
        gc_res_target_to_seed = grangercausalitytests(neural_ts[['target', 'seed']], 1, verbose=False)
        f_diff = gc_res_seed_to_target[1][0]['ssr_ftest'][0] - gc_res_target_to_seed[1][0]['ssr_ftest'][0]
    except Exception as e:
        print(f"Error in GCA analysis {gca_counter}: {str(e)}")
        f_diff = 0
    
    print(f"GCA analysis {gca_counter}: Completed")
    return f_diff

def load_data():
    print('Loading data...')
    all_runs = []
    for run in runs:
        print(f"Loading run {run}")
        try:
            curr_run = image.load_img(f"{raw_dir}/sub-025/ses-01/derivatives/fsl/loc/run-0{run}/1stLevel.feat/filtered_func_data_reg.nii.gz")
            curr_run = image.get_data(image.clean_img(curr_run, standardize=True, mask_img=whole_brain_mask))
            print(f"Run {run} shape: {curr_run.shape}")
            all_runs.append(curr_run)
        except Exception as e:
            print(f"Error loading run {run}: {str(e)}")
        print(f"Memory usage after run {run}: {(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)/1024} MB")
    print('Data loaded. Concatenating...')
    if not all_runs:
        raise ValueError("No valid run data was loaded. Check your input files and paths.")
    bold_vol = np.concatenate(all_runs, axis=3)
    del all_runs
    print(f"Concatenated data shape: {bold_vol.shape}")
    print(f"Final memory usage: {(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)/1024} MB")
    print('Data concatenated...')
    gc.collect()
    return bold_vol

def extract_seed_ts(bold_vol, sub, roi, hemisphere, task='loc', radius=6):
    print("Extracting seed time series...")
    seed_roi_path = f'{study_dir}/{sub}/ses-01/derivatives/rois/spheres_nifti/{sub}_{roi}_{hemisphere}_{task}_sphere_r{radius}mm.nii.gz'
    print(f"Loading seed ROI from: {seed_roi_path}")
    try:
        seed_roi_img = image.load_img(seed_roi_path)
        seed_roi = image.get_data(seed_roi_img)
        print(f"Loaded seed ROI shape: {seed_roi.shape}")
        print(f"Bold volume shape: {bold_vol.shape}")
        if seed_roi.shape[:3] != bold_vol.shape[:3]:
            print("Warning: Seed ROI shape does not match bold volume shape. Attempting to reshape...")
            seed_roi = image.resample_to_img(seed_roi_img, nib.Nifti1Image(bold_vol[:,:,:,0], affine), interpolation='nearest').get_fdata()
            print(f"Reshaped seed ROI to: {seed_roi.shape}")
        
        seed_roi_4d = np.repeat(seed_roi[:,:,:,np.newaxis], bold_vol.shape[-1], axis=3)
        
        masked_img = seed_roi_4d * bold_vol
        seed_ts = masked_img.reshape(-1, bold_vol.shape[-1])
        seed_ts = seed_ts[~np.all(seed_ts == 0, axis=1)]
        seed_ts = np.mean(seed_ts, axis=0)
        print(f"Extracted seed time series shape: {seed_ts.shape}")
        print('Seed data extracted successfully.')
        return seed_ts
    except Exception as e:
        print(f"Error in extract_seed_ts: {str(e)}")
        raise

# Main execution

standard_mask_path = '/user_data/csimmon2/git_repos/ptoc/roiParcels/mruczek_parcels/binary/all_visual_areas.nii.gz'
native_mask_path = transform_mask_to_native(
    f'{exp_dir}/run-01/1stLevel.feat/filtered_func_data_reg.nii.gz',
    standard_mask_path,
    f'{sub_dir}/derivatives/masks'
)
whole_brain_mask = image.load_img(native_mask_path)

affine = whole_brain_mask.affine

print('Searchlight setup ...')
mask = image.get_data(whole_brain_mask)

# Searchlight parameters
sl_rad = 2
max_blk_edge = 10
pool_size = 1
shape = Ball

print(f"Searchlight parameters:")
print(f"Radius: {sl_rad} voxels")
print(f"Max block edge: {max_blk_edge}")
print(f"Pool size: {pool_size}")
print(f"Shape: {shape.__name__}")

bold_vol = load_data()
print(f"bold_vol type: {type(bold_vol)}")
print(f"bold_vol shape: {bold_vol.shape}")
print(f"bold_vol dtype: {bold_vol.dtype}")
seed_ts = extract_seed_ts(bold_vol, sub='sub-025', roi='pIPS', hemisphere='left')

t1 = time.time()
print("Begin Searchlight", print((resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)/1024))
sl = Searchlight(sl_rad=sl_rad, max_blk_edge=max_blk_edge, shape=shape)
print('Distribute', (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)/1024)
sl.distribute([bold_vol], mask)

print('Broadcast', (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)/1024)
sl.broadcast(seed_ts)
print('Run', (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)/1024, flush=True)

# Modify this part to run only a subset of the searchlight
# For example, run only 10 searchlight analyses
max_analyses = 10
sl_result = sl.run_searchlight(gca_searchlight, pool_size=pool_size)
print("End Searchlight\n", (time.time()-t1)/60)

print(f"Searchlight result shape: {sl_result.shape}")
print(f"Non-zero voxels in result: {np.sum(sl_result != 0)}")
print(f"Total GCA analyses performed: {gca_counter}")

sl_result = sl_result.astype('double')
sl_result[np.isnan(sl_result)] = 0
sl_nii = nib.Nifti1Image(sl_result, affine)
nib.save(sl_nii, f'{out_dir}/{study}_sub-025_pIPS_left_gca_subset.nii.gz')