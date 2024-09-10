import os
import pandas as pd
import numpy as np
from nilearn import image, input_data
from nilearn.glm.first_level import compute_regressor
from statsmodels.tsa.stattools import grangercausalitytests
import sys
import nibabel as nib
import logging
from brainiak.searchlight.searchlight import Searchlight
from mpi4py import MPI

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up MPI
comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

# Import parameters
curr_dir = f'/user_data/csimmon2/git_repos/ptoc'
sys.path.insert(0, curr_dir)
import ptoc_params as params

# Set up directories and parameters
study = 'ptoc'
study_dir = f"/lab_data/behrmannlab/vlad/{study}"
results_dir = '/user_data/csimmon2/git_repos/ptoc/results'
raw_dir = params.raw_dir

# Run one subject
subs = ['sub-025']
# Define subjects (adjust as needed)
#sub_info = pd.read_csv(f'{results_dir}/sub_info.csv')
#sub_info = sub_info[sub_info['group'] == 'control']
#subs = [sub for sub in sub_info['sub'].tolist() if sub != 'sub-025']

# Other parameters
rois = ['pIPS', 'LO']
hemispheres = ['left', 'right']
run_num = 3
runs = list(range(1, run_num + 1))
run_combos = [[rn1, rn2] for rn1 in range(1, run_num + 1) for rn2 in range(rn1 + 1, run_num + 1)]

# Helper functions
def standardize_ts(ts):
    return (ts - np.mean(ts)) / np.std(ts)
def check_variance(ts, label, threshold=0.0001):
    variance = np.var(ts)
    if variance < threshold:
        logging.warning(f"Insufficient variance ({variance}) detected for {label} timeseries")
        return False
    return True
def extract_roi_sphere(img, coords):
    roi_masker = input_data.NiftiSpheresMasker([tuple(coords)], radius=6)
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
    
    #creates a binary psychological covariate, previously ran as continuous
    psy[psy > 0] = 1 
    psy[psy < 0] = 0 
    return psy
def extract_cond_ts(ts, cov):
    block_ind = (cov==1)
    block_ind = np.insert(block_ind, 0, True)
    block_ind = np.delete(block_ind, len(block_ind)-1)
    block_ind = (cov == 1).reshape((len(cov))) | block_ind
    return ts[block_ind]

#main analysis
def searchlight_gca(data, mask, bcvar, myrad):
    # Extract the time series for the current searchlight sphere
    sphere_ts = data[0]
    
    # Get the mask for the current sphere
    sphere_mask = mask[0]
    
    # Count the number of voxels in the sphere
    n_voxels = np.sum(sphere_mask)
    
    # Set minimum voxels threshold (adjust as needed for 8-10mm radius)
    min_voxels = 30  # Adjust based on your searchlight size
    
    # Check if the sphere has enough voxels
    if n_voxels < min_voxels:
        return np.nan
    
    # Extract the time series for the comparison region (e.g., pIPS)
    comparison_ts = bcvar['comparison_ts']
    
    # Extract the psychological covariate
    psy = bcvar['psy']
    
    # Perform condition-specific extraction
    sphere_phys = extract_cond_ts(sphere_ts, psy)
    comparison_phys = extract_cond_ts(comparison_ts, psy)
    
    # Standardize the time series
    sphere_phys_standardized = standardize_ts(sphere_phys)
    comparison_phys_standardized = standardize_ts(comparison_phys)
    
    # Check for sufficient variance in the data
    if not check_variance(sphere_phys_standardized, "Sphere") or \
       not check_variance(comparison_phys_standardized, "Comparison ROI"):
        return np.nan
    
    # Perform Granger causality tests
    neural_ts = pd.DataFrame({
        'sphere': sphere_phys_standardized.ravel(),
        'comparison': comparison_phys_standardized.ravel()
    })
    
    # Perform bidirectional Granger causality tests
    gc_res_sphere_to_comparison = grangercausalitytests(neural_ts[['sphere', 'comparison']], 1, verbose=False)
    gc_res_comparison_to_sphere = grangercausalitytests(neural_ts[['comparison', 'sphere']], 1, verbose=False)
    
    # Calculate the difference in F-statistics
    f_sphere_to_comparison = gc_res_sphere_to_comparison[1][0]['ssr_ftest'][0]
    f_comparison_to_sphere = gc_res_comparison_to_sphere[1][0]['ssr_ftest'][0]
    f_diff = f_sphere_to_comparison - f_comparison_to_sphere
    
    return f_diff

# partial script for testing:
def conduct_mini_searchlight_gca():
    logging.info('Running Searchlight GCA...')
    tasks = ['loc']
    
    for ss in subs:
        sub_dir = f'{study_dir}/{ss}/ses-01/'
        temp_dir = f'{raw_dir}/{ss}/ses-01'
        roi_dir = f'{sub_dir}/derivatives/rois'
        exp_dir = f'{temp_dir}/derivatives/fsl/loc'
        os.makedirs(f'{sub_dir}/derivatives/results/searchlight_gca', exist_ok=True)

        roi_coords = pd.read_csv(f'{roi_dir}/spheres/sphere_coords_hemisphere.csv')
        logging.info(f"ROI coordinates loaded for subject {ss}")

        # Select only one run combination
        rc = run_combos[0]  # This will use the first run combination
        logging.info(f"Processing run combination {rc} for subject {ss}")
        
        # 1. Prepare the data (4D volume)
        filtered_list = []
        for rn in rc:
            curr_run = image.load_img(f'{exp_dir}/run-0{rn}/1stLevel.feat/filtered_func_data_reg.nii.gz')
            curr_run = image.clean_img(curr_run, standardize=True)
            filtered_list.append(curr_run)

        img4d = image.concat_imgs(filtered_list)
        data = img4d.get_fdata()
        logging.info(f"Data shape: {data.shape}")

        # 2. Prepare the mask (3D binary mask)
        whole_brain_mask = image.math_img("np.any(img > 0, axis=-1)", img=img4d)
        mask = whole_brain_mask.get_fdata().astype(bool)
        logging.info(f"Mask shape: {mask.shape}")

        for tsk in tasks:
            # Select only one comparison ROI and hemisphere
            comparison_roi = 'pIPS'
            comparison_hemi = 'left'
            
            comparison_coords = roi_coords[(roi_coords['index'] == 0) & 
                                           (roi_coords['task'] == tsk) & 
                                           (roi_coords['roi'] == comparison_roi) &
                                           (roi_coords['hemisphere'] == comparison_hemi)]
            
            if comparison_coords.empty:
                logging.warning(f"No coordinates found for {comparison_roi}, {comparison_hemi}, run combo {rc}")
                continue

            comparison_ts = extract_roi_sphere(img4d, comparison_coords[['x', 'y', 'z']].values.tolist()[0])
            
            psy = make_psy_cov(rc, ss)
            
            # 3. Prepare bcvar (dictionary with additional variables)
            bcvar = {'comparison_ts': comparison_ts, 'psy': psy}

            # 4. Set searchlight parameters
            sl_rad = 3  # This gives approximately 9 mm radius for 3 mm isotropic voxels
            max_blk_edge = 5  # Adjust as needed
            pool_size = 1

            # Set up the searchlight
            sl = Searchlight(sl_rad=sl_rad, max_blk_edge=max_blk_edge)

            # Distribute data to the searchlights
            sl.distribute([data], mask)

            # Broadcast the additional variables
            sl.broadcast(bcvar)

            # Run the searchlight
            sl_result = sl.run_searchlight(searchlight_gca, pool_size=pool_size)

            # Save the searchlight results
            sl_img = nib.Nifti1Image(sl_result, img4d.affine, img4d.header)
            output_file = f'{sub_dir}/derivatives/results/searchlight_gca/sub-{ss}_task-{tsk}_comp-{comparison_roi}_{comparison_hemi}_searchlight_gca.nii.gz'
            nib.save(sl_img, output_file)
            logging.info(f"Saved searchlight results to {output_file}")

        logging.info(f'Completed Searchlight GCA for subject {ss}')

if __name__ == "__main__":
    conduct_mini_searchlight_gca()
    
#full script for all subs
def conduct_searchlight_gca():
    logging.info('Running Searchlight GCA...')
    tasks = ['loc']
    
    for ss in subs:
        sub_dir = f'{study_dir}/{ss}/ses-01/'
        temp_dir = f'{raw_dir}/{ss}/ses-01'
        roi_dir = f'{sub_dir}/derivatives/rois'
        exp_dir = f'{temp_dir}/derivatives/fsl/loc'
        os.makedirs(f'{sub_dir}/derivatives/results/searchlight_gca', exist_ok=True)

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

            # Create a whole-brain mask
            whole_brain_mask = image.math_img("img > 0", img=img4d.mean_img())

            for tsk in tasks:
                for comparison_roi in ['pIPS']:
                    for comparison_hemi in hemispheres:
                        comparison_coords = roi_coords[(roi_coords['index'] == rcn) & 
                                                       (roi_coords['task'] == tsk) & 
                                                       (roi_coords['roi'] == comparison_roi) &
                                                       (roi_coords['hemisphere'] == comparison_hemi)]
                        
                        if comparison_coords.empty:
                            logging.warning(f"No coordinates found for {comparison_roi}, {comparison_hemi}, run combo {rc}")
                            continue

                        comparison_ts = extract_roi_sphere(img4d, comparison_coords[['x', 'y', 'z']].values.tolist()[0])
                        
                        psy = make_psy_cov(rc, ss)
                        
                        # Set up the searchlight
                        sl_rad = 3  # Adjust this value as needed (3 voxels = ~9mm radius)
                        max_blk_edge = 5  # Adjust as needed
                        pool_size = 1
                        sl = Searchlight(sl_rad=sl_rad, max_blk_edge=max_blk_edge)

                        # Distribute data to the searchlights
                        sl.distribute([img4d.get_fdata()], whole_brain_mask.get_fdata())

                        # Broadcast the comparison time series and psychological covariate
                        sl.broadcast({'comparison_ts': comparison_ts, 'psy': psy})

                        # Run the searchlight
                        sl_result = sl.run_searchlight(searchlight_gca, pool_size=pool_size)

                        # Save the searchlight results
                        sl_img = nib.Nifti1Image(sl_result, img4d.affine, img4d.header)
                        output_file = f'{sub_dir}/derivatives/results/searchlight_gca/sub-{ss}_task-{tsk}_comp-{comparison_roi}_{comparison_hemi}_searchlight_gca.nii.gz'
                        nib.save(sl_img, output_file)
                        logging.info(f"Saved searchlight results to {output_file}")

        logging.info(f'Completed Searchlight GCA for subject {ss}')

#if __name__ == "__main__":
    #conduct_searchlight_gca()