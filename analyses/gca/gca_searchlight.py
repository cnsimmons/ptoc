import warnings
warnings.filterwarnings("ignore")
import resource
import sys
import time
import os
import gc
import pandas as pd
import numpy as np
import pdb

from sklearn.decomposition import PCA
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LinearRegression

from nilearn import image, datasets
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
localizer = 'Scramble' # scramble or object. This is the localizer task.
results_dir = '/user_data/csimmon2/git_repos/ptoc/results'
raw_dir = params.raw_dir

# Load subject information
sub_info = pd.read_csv(f'{curr_dir}/sub_info.csv')
sub_info = sub_info[sub_info['group'] == 'control']
#subs = sub_info['sub'].tolist()
subs = ['sub-025']

# Other parameters
rois = ['pIPS', 'LO']
hemispheres = ['left', 'right']
run_num = 3
runs = list(range(1, run_num + 1))
run_combos = [[rn1, rn2] for rn1 in range(1, run_num + 1) for rn2 in range(rn1 + 1, run_num + 1)]

whole_brain_mask = image.load_img('/user_data/csimmon2/git_repos/ptoc/roiParcels/mruczek_parcels/binary/all_visual_areas.nii.gz')
affine = whole_brain_mask.affine
dimsizes = whole_brain_mask.header.get_zooms() # get dimensions

# scan parameters
vols = 184
tr = 2.0
#first_fix = 8 #I'm not sure if I should change this for my data - will need to run by Vlad

'''
Set up Searchlight
'''

print ("Setting up searchlight...")
mask = image.get_data(whole_brain_mask) #the mask to search within
sl_rad = 2 #searchlight radius in voxels
max_blk_edge = 10 #how many blocks to send on each parallelized search
pool_size = 1 #how many cores to use

voxels_proprotion = 1
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
    psy[psy > 0] = 1 #remove if run my approach
    psy[psy < 0] = 0 #remove if run my approach
    return psy

def extract_cond_ts(ts, cov):
    block_ind = (cov==1)
    block_ind = np.insert(block_ind, 0, True)
    block_ind = np.delete(block_ind, len(block_ind)-1)
    block_ind = (cov == 1).reshape((len(cov))) | block_ind
    return ts[block_ind]

def gca(data, sl_mask, myrad, seed_ts):
    logging.info(f'Running GCA for {localizer}...')
    tasks = ['loc']
    
    #pull out data
    data4d = data[0]
    data4d = np.transpose(data4d.reshape (-1,data[0].shape[3]))

    # Perform condition-specific extraction
    sphere_phys = extract_cond_ts(sphere_ts, psy)
    comparison_phys = extract_cond_ts(comparison_ts, psy)

    # Extract the time series for the current searchlight sphere
    sphere_ts = data[0]
    
    # Get the mask for the current sphere
    sphere_mask = mask[0] if isinstance(mask, np.ndarray) and mask.ndim > 2 else mask

    for ss in subs:
        sub_summary = pd.DataFrame(columns=['sub', 'fold', 'task', 'origin', 'target', 'f_diff'])
        
        sub_dir = f'{study_dir}/{ss}/ses-01/'
        temp_dir = f'{raw_dir}/{ss}/ses-01'
        roi_dir = f'{sub_dir}/derivatives/rois'
        exp_dir = f'{temp_dir}/derivatives/fsl/loc'
        os.makedirs(f'{sub_dir}/derivatives/gca', exist_ok=True)

        roi_coords = pd.read_csv(f'{roi_dir}/spheres/sphere_coords_hemisphere_{localizer.lower()}.csv') #remove _{localizer} to run object
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
        sub_summary.to_csv(f'{sub_dir}/derivatives/gca/gca_searchlight.csv', index=False)
        
        
def load_data():
    print('Loading data...')

    all_runs = []
    for run in runs:
        print(run)

        curr_run = image.load_img(f'{exp_dir}/run-0{run}/1stLevel.feat/filtered_func_data_reg.nii.gz') #load data
        curr_run = image.get_data(image.clean_img(curr_run,standardize=True,mask_img=whole_brain_mask)) #standardize within mask and convert to numpy
        #curr_run = curr_run[:,:,:,first_fix:] #remove first few fixation volumes

        all_runs.append(curr_run) #append to list


        del curr_run
        print((resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)/1024)

    print('data loaded..')

    print('concatenating data..')
    bold_vol = np.concatenate(np.array(all_runs),axis = 3) #compile into 4D
    del all_runs
    print((resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)/1024)
    print('data concatenated...')
    gc.collect()

    return bold_vol

def extract_seed_ts(bold_vol):
    """
    extract all data from seed region
    """

    #load seed
    seed_roi = image.get_data(image.load_img(f'{roi_dir}/spheres/{dorsal}_sphere.nii.gz'))
    reshaped_roi = np.reshape(seed_roi, (91,109,91,1))
    masked_img = reshaped_roi*bold_vol

    #extract voxel resposnes from within mask
    seed_ts = masked_img.reshape(-1, bold_vol.shape[3]) #reshape into rows (voxels) x columns (time)
    seed_ts =seed_ts[~np.all(seed_ts == 0, axis=1)] #remove voxels that are 0 (masked out)
    seed_ts = np.transpose(seed_ts)

    print('Seed data extracted...')

    return seed_ts


bold_vol = load_data()
seed_ts = extract_seed_ts(bold_vol)

#run searchlight
t1 = time.time()
print("Begin Searchlight", print((resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)/1024))
sl = Searchlight(sl_rad=sl_rad,max_blk_edge=max_blk_edge, shape = shape) #setup the searchlight
print('Distribute', (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)/1024)
sl.distribute([bold_vol], mask) #send the 4dimg and mask

print('Broadcast', (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)/1024)
sl.broadcast(seed_ts) #send the relevant analysis vars
print('Run', (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)/1024, flush= True)
sl_result = sl.run_searchlight(gca, pool_size=pool_size)
print("End Searchlight\n", (time.time()-t1)/60)
        