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

# Other parameters
rois = ['pIPS', 'LO']
hemispheres = ['left', 'right']
run_num = 3
runs = list(range(1, run_num + 1))
run_combos = [[rn1, rn2] for rn1 in range(1, run_num + 1) for rn2 in range(rn1 + 1, run_num + 1)]

# Helper functions (keep these as they were)

def searchlight_gca(data, mask, bcvar, myrad):
    logging.info(f"searchlight_gca called with: data shape: {data[0].shape if isinstance(data, list) else data.shape}")
    logging.info(f"mask shape: {mask.shape}")
    logging.info(f"bcvar type: {type(bcvar)}")
    logging.info(f"myrad: {myrad}")

    try:
        # Extract the time series for the current searchlight sphere
        sphere_ts = data[0]
        
        # Get the mask for the current sphere
        sphere_mask = mask[0] if isinstance(mask, np.ndarray) and mask.ndim > 2 else mask
        
        # Count the number of voxels in the sphere
        n_voxels = np.sum(sphere_mask)
        
        # Set minimum voxels threshold
        min_voxels = 30
        
        # Check if the sphere has enough voxels
        if n_voxels < min_voxels:
            return np.nan
        
        # Extract the time series for the comparison region and the psychological covariate
        comparison_ts = bcvar['comparison_ts']
        psy = bcvar['psy']
        
        # Perform condition-specific extraction
        sphere_phys = extract_cond_ts(sphere_ts, psy)
        comparison_phys = extract_cond_ts(comparison_ts, psy)
        
        # Ensure both arrays have the same length
        min_length = min(len(sphere_phys), len(comparison_phys))
        sphere_phys = sphere_phys[:min_length]
        comparison_phys = comparison_phys[:min_length]

        # Perform Granger causality tests
        neural_ts = pd.DataFrame({
            'sphere': sphere_phys.ravel(),
            'comparison': comparison_phys.ravel()
        })
        
        gc_res_sphere_to_comparison = grangercausalitytests(neural_ts[['sphere', 'comparison']], 1, verbose=False)
        gc_res_comparison_to_sphere = grangercausalitytests(neural_ts[['comparison', 'sphere']], 1, verbose=False)
        
        f_sphere_to_comparison = gc_res_sphere_to_comparison[1][0]['ssr_ftest'][0]
        f_comparison_to_sphere = gc_res_comparison_to_sphere[1][0]['ssr_ftest'][0]
        f_diff = f_sphere_to_comparison - f_comparison_to_sphere
    except Exception as e:
        logging.error(f"Error in searchlight_gca: {str(e)}")
        return np.nan
    
    return f_diff

def conduct_mini_searchlight_gca():
    logging.info('Running Mini Searchlight GCA...')
    tasks = ['loc']
    
    for ss in subs:
        try:
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
            
            # Load and preprocess the data
            filtered_list = []
            for rn in rc:
                curr_run = image.load_img(f'{exp_dir}/run-0{rn}/1stLevel.feat/filtered_func_data_reg.nii.gz')
                curr_run = image.clean_img(curr_run, standardize=False)
                filtered_list.append(curr_run)

            # Concatenate runs if more than one
            if len(filtered_list) > 1:
                img4d = image.concat_imgs(filtered_list)
            else:
                img4d = filtered_list[0]

            # Get the data as a 4D numpy array
            data = img4d.get_fdata()
            logging.info(f"Data shape: {data.shape}")

            # Create a whole-brain mask
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
                
                # Prepare bcvar (dictionary with additional variables)
                bcvar = {'comparison_ts': comparison_ts.ravel(), 'psy': psy.ravel()}
                logging.info(f"bcvar keys: {bcvar.keys()}")
                logging.info(f"comparison_ts shape: {bcvar['comparison_ts'].shape}")
                logging.info(f"psy shape: {bcvar['psy'].shape}")

                # Set up the searchlight
                sl_rad = 3
                max_blk_edge = 5
                pool_size = 1
                sl = Searchlight(sl_rad=sl_rad, max_blk_edge=max_blk_edge)
                logging.info(f"Searchlight initialized with sl_rad={sl_rad}, max_blk_edge={max_blk_edge}")

                # Distribute data to the searchlights
                sl.distribute([data], mask)
                logging.info("Data distributed to searchlights")

                # Broadcast the additional variables
                sl.broadcast(bcvar)
                logging.info("Additional variables broadcasted")

                # Run the searchlight
                logging.info("Starting searchlight analysis...")
                sl_result = sl.run_searchlight(searchlight_gca, pool_size=pool_size)
                logging.info("Searchlight analysis completed")
                logging.info(f"Searchlight result shape: {sl_result.shape}")

                # Save the searchlight results
                sl_img = nib.Nifti1Image(sl_result, img4d.affine, img4d.header)
                output_file = f'{sub_dir}/derivatives/results/searchlight_gca/sub-{ss}_task-{tsk}_comp-{comparison_roi}_{comparison_hemi}_searchlight_gca.nii.gz'
                nib.save(sl_img, output_file)
                logging.info(f"Saved searchlight results to {output_file}")

            logging.info(f'Completed Searchlight GCA for subject {ss}')

        except Exception as e:
            logging.error(f"Error processing subject {ss}: {str(e)}")

    logging.info('Mini Searchlight GCA completed for all subjects')

if __name__ == "__main__":
    try:
        conduct_mini_searchlight_gca()
    except Exception as e:
        logging.error(f"Fatal error in main script execution: {str(e)}")