import os
import pandas as pd
import numpy as np
from nilearn import image, input_data
from nilearn.glm.first_level import compute_regressor
from statsmodels.tsa.stattools import grangercausalitytests
import nibabel as nib
import logging
import sys
from brainiak.searchlight.searchlight import Searchlight, Ball 
from mpi4py import MPI
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Set up searchlight parameters
sl_rad = 2 #search light radius
max_blk_edge = 10 #max block edge
pool_size = 1
voxels_proportion = 1
shape = Ball

# Import your parameters
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
subs = sub_info['sub'].tolist()
#subs = ['sub-057']

rois = ['pIPS', 'LO']
hemispheres = ['left', 'right']
run_num = 3
runs = list(range(1, run_num + 1))
run_combos = [[rn1, rn2] for rn1 in range(1, run_num + 1) for rn2 in range(rn1 + 1, run_num + 1)]


# Load whole brain mask
whole_brain_mask = image.load_img('roiParcels/mruczek_parcels/binary/all_visual_areas.nii.gz') 
mask = image.get_data(whole_brain_mask) # the mask to search within 

def extract_roi_sphere(img, coords):
    roi_masker = input_data.NiftiSpheresMasker([tuple(coords)], radius=6)
    seed_time_series = roi_masker.fit_transform(img)
    return np.mean(seed_time_series, axis=1).reshape(-1, 1)

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
    psy[psy <= 0] = 0
    return psy

def extract_cond_ts(ts, cov):
    block_ind = (cov==1)
    block_ind = np.insert(block_ind, 0, True)
    block_ind = np.delete(block_ind, len(block_ind)-1)
    block_ind = (cov == 1).reshape((len(cov))) | block_ind
    return ts[block_ind]

def searchlight_gca(data, sl_mask, myrad, bcvar):
    sphere_ts = np.mean(data[0], axis=0)
    seed_ts, psy = bcvar
    
    sphere_phys = extract_cond_ts(sphere_ts, psy)
    seed_phys = extract_cond_ts(seed_ts.ravel(), psy)
    
    if len(sphere_phys) < 2 or len(seed_phys) < 2:
        return np.nan
    
    neural_ts = pd.DataFrame({
        'seed': seed_phys,
        'sphere': sphere_phys
    })
    
    try:
        gc_res_seed_to_sphere = grangercausalitytests(neural_ts[['seed', 'sphere']], 1, verbose=False)
        gc_res_sphere_to_seed = grangercausalitytests(neural_ts[['sphere', 'seed']], 1, verbose=False)
        
        f_seed_to_sphere = gc_res_seed_to_sphere[1][0]['ssr_ftest'][0]
        f_sphere_to_seed = gc_res_sphere_to_seed[1][0]['ssr_ftest'][0]
        
        f_diff = f_seed_to_sphere - f_sphere_to_seed
    except:
        f_diff = np.nan
    
    return f_diff

def conduct_gca():
    logging.info('Running GCA...')
    tasks = ['loc']
    
    for ss in subs:
        sub_dir = f'{study_dir}/{ss}/ses-01/'
        temp_dir = f'{raw_dir}/{ss}/ses-01'
        roi_dir = f'{sub_dir}/derivatives/rois'
        exp_dir = f'{temp_dir}/derivatives/fsl/loc'
        os.makedirs(f'{sub_dir}/derivatives/gca', exist_ok=True)

        roi_coords = pd.read_csv(f'{roi_dir}/spheres/sphere_coords_hemisphere.csv')
        logging.info(f"ROI coordinates loaded for subject {ss}")

        for tsk in tasks:
            # Load and preprocess all runs
            all_runs = []
            for run in runs:
                curr_run = image.load_img(f'{exp_dir}/run-0{run}/1stLevel.feat/filtered_func_data_reg.nii.gz')
                curr_run = image.clean_img(curr_run, standardize=True)
                all_runs.append(curr_run)
            
            img4d = image.concat_imgs(all_runs)
            logging.info(f"Concatenated image shape: {img4d.shape}")

            psy = make_psy_cov(runs, ss)
            
            for seed_roi in rois:
                for seed_hemi in hemispheres:
                    seed_coords = roi_coords[(roi_coords['task'] == tsk) & 
                                             (roi_coords['roi'] == seed_roi) &
                                             (roi_coords['hemisphere'] == seed_hemi)]
                    
                    if seed_coords.empty:
                        logging.warning(f"No coordinates found for {seed_roi}_{seed_hemi}")
                        continue

                    seed_ts = extract_roi_sphere(img4d, seed_coords[['x', 'y', 'z']].values.tolist()[0])
                    
                    if seed_ts.shape[0] != psy.shape[0]:
                        raise ValueError(f"Mismatch in volumes: {seed_roi}_ts has {seed_ts.shape[0]}, psy has {psy.shape[0]}")
                    
                    # Set up and run searchlight
                    sl = Searchlight(sl_rad=sl_rad, max_blk_edge=max_blk_edge)
                    data = img4d.get_fdata()
                    
                    sl.distribute([data], mask)
                    sl.broadcast((seed_ts, psy))
                    
                    sl_result = sl.run_searchlight(searchlight_gca, pool_size=pool_size)
                    
                    # Save searchlight results
                    sl_img = nib.Nifti1Image(sl_result, img4d.affine)
                    output_file = f'{sub_dir}/derivatives/gca/sub-{ss}_task-{tsk}_seed-{seed_roi}_{seed_hemi}_searchlight_gca.nii.gz'
                    nib.save(sl_img, output_file)
                    
                    logging.info(f"Completed Searchlight GCA for {ss}, {tsk}, {seed_roi}_{seed_hemi}")

        logging.info(f'Completed Searchlight GCA for subject {ss}')

##NEW##

def load_data():
    print('Loading data...')

    all_runs = []
    for run in runs:
        print(run)

        curr_run = image.load_img(f'{exp_dir}/run-0{run}/1stLevel.feat/filtered_func_data_reg.nii.gz') #load data
        curr_run = image.get_data(image.clean_img(curr_run,standardize=True,mask_img=whole_brain_mask)) #standardize within mask and convert to numpy
        curr_run = curr_run[:,:,:,first_fix:] #remove first few fixation volumes

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
sl_result = sl.run_searchlight(mvpd, pool_size=pool_size)
print("End Searchlight\n", (time.time()-t1)/60)

sl_result = sl_result.astype('double')  # Convert the output into a precision format that can be used by other applications
sl_result[np.isnan(sl_result)] = 0  # Exchange nans with zero to ensure compatibility with other applications
sl_nii = nib.Nifti1Image(sl_result, affine)  # create the volume image
nib.save(sl_nii, f'{out_dir}/{study}{ss}_{dorsal}_searchlight.nii.gz')  # Save the volume

if __name__ == "__main__":
    conduct_gca()