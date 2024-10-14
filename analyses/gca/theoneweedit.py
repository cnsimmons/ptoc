## GCA with localizer specified ##

            #img4d = image.concat_imgs(filtered_list)
            #logging.info(f"Concatenated image shape: {img4d.shape}")

            #if len(brain_masks) > 1:
                #combined_mask_data = np.all([mask.get_fdata() > 0 for mask in brain_masks], axis=0)
                #combined_brain_mask = nib.Nifti1Image(combined_mask_data.astype(np.int32), brain_masks[0].affine)
            #else:
                #combined_brain_mask = brain_masks[0]

                    # Create searchlight object
                    #searchlight = SearchLight(
                        #combined_brain_mask, 
                        #process_mask_img=combined_brain_mask, 
                        #radius=6,
                        #n_jobs=-1,  # Use all available cores
                        #verbose=0
                    #)
                                        
## GCA with localizer specified using Brainiak ##
import sys
import os
import logging
import pandas as pd
import numpy as np
from nilearn import image, maskers
from nilearn.glm.first_level import compute_regressor
from statsmodels.tsa.stattools import grangercausalitytests
import nibabel as nib
from tqdm import tqdm
from brainiak.searchlight.searchlight import Ball
from mpi4py import MPI

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import your parameters
curr_dir = f'/user_data/csimmon2/git_repos/ptoc'
sys.path.insert(0, curr_dir)
import ptoc_params as params

# Set up directories and parameters
study = 'ptoc'
study_dir = f"/lab_data/behrmannlab/vlad/{study}"
localizer = 'Object'
results_dir = '/user_data/csimmon2/git_repos/ptoc/results'
raw_dir = params.raw_dir

# Load subject information
sub_info = pd.read_csv(f'{curr_dir}/sub_info.csv')
sub_info = sub_info[sub_info['group'] == 'control']
subs = ['sub-025']
rois = ['pIPS']
hemispheres = ['left']

run_num = 3
runs = list(range(1, run_num + 1))
run_combos = [[rn1, rn2] for rn1 in range(1, run_num + 1) for rn2 in range(rn1 + 1, run_num + 1)]

def extract_roi_sphere(img, coords):
    roi_masker = maskers.NiftiSpheresMasker([tuple(coords)], radius=6)
    seed_time_series = roi_masker.fit_transform(img)
    phys = np.mean(seed_time_series, axis=1).reshape(-1, 1)
    return phys

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

def gca_function(data, roi_ts):
    sphere_ts = np.mean(data, axis=0).reshape(-1, 1)
    
    try:
        neural_ts = pd.DataFrame({'roi': roi_ts.ravel(), 'sphere': sphere_ts.ravel()})
        gc_res_sphere_to_roi = grangercausalitytests(neural_ts[['sphere', 'roi']], 1, verbose=False)
        gc_res_roi_to_sphere = grangercausalitytests(neural_ts[['roi', 'sphere']], 1, verbose=False)
        f_diff = gc_res_sphere_to_roi[1][0]['ssr_ftest'][0] - gc_res_roi_to_sphere[1][0]['ssr_ftest'][0]
    except Exception as e:
        logging.warning(f"Error in GCA calculation: {str(e)}")
        f_diff = 0
    
    return f_diff

def conduct_gca():
    logging.info(f'Running GCA for {localizer}...')
    
    for ss in tqdm(subs, desc="Processing subjects"):
        sub_summary = pd.DataFrame(columns=['sub', 'roi', 'hemisphere', 'task', 'x', 'y', 'z', 'f_diff'])
        
        sub_dir = f'{study_dir}/{ss}/ses-01/'
        temp_dir = f'{raw_dir}/{ss}/ses-01'
        roi_dir = f'{sub_dir}/derivatives/rois'
        exp_dir = f'{temp_dir}/derivatives/fsl/loc'
        os.makedirs(f'{sub_dir}/derivatives/gca_searchlight', exist_ok=True)

        roi_coords = pd.read_csv(f'{roi_dir}/spheres/sphere_coords_hemisphere.csv')
        logging.info(f"ROI coordinates loaded for subject {ss}")

        for rcn, rc in enumerate(run_combos):
            logging.info(f"Processing run combination {rc} for subject {ss}")
            
            filtered_list = []
            for rn in rc:
                curr_run = image.load_img(f'{exp_dir}/run-0{rn}/1stLevel.feat/filtered_func_data_reg.nii.gz')
                curr_run = image.clean_img(curr_run, standardize='zscore_sample')
                filtered_list.append(curr_run)
                
            img4d = image.concat_imgs(filtered_list)
            logging.info(f"Concatenated image shape: {img4d.shape}")

            # Create a small test mask (3x3x3 cube)
            test_mask_shape = (10, 10, 10)  # Adjust size as needed
            test_mask_data = np.zeros(test_mask_shape)
            center = np.array(test_mask_shape) // 2
            test_mask_data[center[0]-1:center[0]+2, center[1]-1:center[1]+2, center[2]-1:center[2]+2] = 1

            # Create a NIfTI image from the test mask data
            affine = img4d.affine
            test_mask_nifti = nib.Nifti1Image(test_mask_data, affine)

            # Crop the 4D data to match the test mask size
            img4d_data = img4d.get_fdata()
            cropped_data = img4d_data[:test_mask_shape[0], :test_mask_shape[1], :test_mask_shape[2], :]

            for roi in tqdm(rois, desc="Processing ROIs", leave=False):
                for hemisphere in tqdm(hemispheres, desc="Processing hemispheres", leave=False):
                    logging.info(f"Processing {hemisphere} {roi}")
                    
                    roi_coord = roi_coords[(roi_coords['index'] == rcn) & 
                                        (roi_coords['task'] == 'loc') & 
                                        (roi_coords['roi'] == roi) &
                                        (roi_coords['hemisphere'] == hemisphere)].iloc[0]
                    
                    roi_ts = extract_roi_sphere(img4d, roi_coord[['x', 'y', 'z']].values.tolist())
                    
                    # Prepare data for Brainiak searchlight
                    data = cropped_data.transpose((3, 0, 1, 2))

                    # Set up the searchlight parameters
                    comm = MPI.COMM_WORLD
                    rank = comm.rank
                    size = comm.size

                    # Create and run Brainiak searchlight
                    sl_rad = 1  # Small radius for test mask
                    ball = Ball(sl_rad)
                    sl_mask = test_mask_data.astype(bool)

                    # Define the searchlight function
                    def brainiak_gca(data, mask):
                        if not np.any(mask):
                            return 0
                        f_diff = gca_function(data, roi_ts)
                        return f_diff

                    logging.info(f"Starting Brainiak searchlight analysis for {hemisphere} {roi}...")
                    sl_result = ball.run(data, sl_mask, brainiak_gca)

                    # Gather results from all processes
                    gathered_results = comm.gather(sl_result, root=0)

                    if rank == 0:
                        # Combine results
                        searchlight_results = np.zeros(test_mask_shape)
                        for result in gathered_results:
                            searchlight_results += result

                        # Save searchlight results
                        logging.info("Saving searchlight results...")
                        searchlight_img = nib.Nifti1Image(searchlight_results, affine)
                        output_path = f'{sub_dir}/derivatives/gca_searchlight/test_brainiak_searchlight_results_{hemisphere}_{roi}_{rc[0]}-{rc[-1]}.nii.gz'
                        nib.save(searchlight_img, output_path)
                        logging.info(f"Searchlight results saved to: {output_path}")

                        # Extract results for summary
                        logging.info("Extracting results for summary...")
                        non_zero_indices = np.nonzero(searchlight_results)

                        for x, y, z in zip(*non_zero_indices):
                            f_diff = searchlight_results[x, y, z]
                            curr_data = pd.Series({
                                'sub': ss,
                                'roi': roi,
                                'hemisphere': hemisphere,
                                'x': x,
                                'y': y,
                                'z': z,
                                'task': 'loc',
                                'f_diff': f_diff
                            })
                            sub_summary = sub_summary.append(curr_data, ignore_index=True)

        logging.info(f'Completed GCA searchlight for subject {ss}')
        summary_path = f'{sub_dir}/derivatives/gca_searchlight/brainiak_gca_searchlight_summary_{localizer.lower()}.csv'
        sub_summary.to_csv(summary_path, index=False)
        logging.info(f"Summary saved to: {summary_path}")
        
if __name__ == "__main__":
    conduct_gca()