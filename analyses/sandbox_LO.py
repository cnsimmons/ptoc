import os
import pandas as pd
import numpy as np
from nilearn import image, maskers, plotting
from nilearn.maskers import NiftiMasker
from nilearn.datasets import load_mni152_brain_mask, load_mni152_template
from nilearn import datasets
from nilearn.glm.first_level import compute_regressor
import nibabel as nib
import sys
import time
import itertools 

# Import your parameters
curr_dir = f'/user_data/csimmon2/git_repos/ptoc'
sys.path.insert(0, curr_dir)
import ptoc_params as params

# Set up directories and parameters
study = 'ptoc'
study_dir = f"/lab_data/behrmannlab/vlad/{study}"
ptoc_dir = f"/lab_data/behrmannlab/vlad/{study}"
results_dir = f'/user_data/csimmon2/git_repos/ptoc/results'
hemispace_dir = f'/lab_data/behrmannlab/vlad/hemispace' 
raw_dir = params.raw_dir

sub_info = pd.read_csv(f'{curr_dir}/sub_info.csv')
#subs = sub_info[sub_info['group'] == 'control']['sub'].tolist()
subs = ['sub-025']
rois = ['LO', 'pIPS']

'''scan params'''
tr = 2.0
vols = 184

whole_brain_mask = load_mni152_brain_mask()
mni = load_mni152_template()
brain_masker = NiftiMasker(whole_brain_mask, smoothing_fwhm=0, standardize=True)

#run combos
run_num = 3
runs = list(range(1,run_num+1))
run_combos = []

#determine the number of left out run combos
for rn1 in range(1,run_num+1):
    for rn2 in range(rn1+1,run_num+1):
        run_combos.append([rn1,rn2])

def extract_roi_sphere(img, coords, radius=6):
    roi_masker = maskers.NiftiSpheresMasker([tuple(coords)], radius=radius)
    seed_time_series = roi_masker.fit_transform(img)
    phys = np.mean(seed_time_series, axis=1)
    
    phys = phys.reshape((phys.shape[0],1))
    
    return phys

def make_psy_cov(runs, ss):
    temp_dir = f'{raw_dir}/{ss}/ses-01'
    cov_dir = f'{temp_dir}/covs'
    vols, tr = 184, 2.0
    times = np.arange(0, vols * len(runs) * tr, tr)
    full_cov = pd.DataFrame(columns=['onset', 'duration', 'value'])
    
    for rn, run in enumerate(runs):
        ss_num = ss.split('-')[1]
        obj_cov_file = f'{cov_dir}/catloc_{ss_num}_run-0{run}_Object.txt'
        scr_cov_file = f'{cov_dir}/catloc_{ss_num}_run-0{run}_Scramble.txt'
        
        if not os.path.exists(obj_cov_file) or not os.path.exists(scr_cov_file):
            print(f'Covariate file not found for run {run}')
            raise FileNotFoundError(f"Missing covariate file for run {run}")
        
        obj_cov = pd.read_csv(obj_cov_file, sep='\t', header=None, names=['onset', 'duration', 'value'])
        scr_cov = pd.read_csv(scr_cov_file, sep='\t', header=None, names=['onset', 'duration', 'value'])
        scr_cov['value'] *= -1
        
        curr_cov = pd.concat([obj_cov, scr_cov])
        curr_cov['onset'] += rn * vols * tr
        full_cov = pd.concat([full_cov, curr_cov])
    
    full_cov = full_cov.sort_values(by=['onset']).reset_index(drop=True)
    
    if full_cov['onset'].max() >= times[-1] or full_cov['onset'].min() < 0:
        raise ValueError("Event onsets are outside the expected time range")
    
    cov = full_cov.to_numpy()
    
    psy, _ = compute_regressor(cov.T, 'spm', times)
    
    if len(psy) != vols * len(runs):
        raise ValueError(f"Generated psy does not match expected length. Expected {vols * len(runs)}, got {len(psy)}")
    
    return psy

def calculate_fc(brain_time_series, phys):
    min_length = min(brain_time_series.shape[0], phys.shape[0])
    brain_time_series = brain_time_series[:min_length]
    phys = phys[:min_length].ravel()

    print(f"In calculate_fc:")
    print(f"  Brain time series shape: {brain_time_series.shape}")
    print(f"  Phys shape: {phys.shape}")

    constant_voxels = np.where(np.std(brain_time_series, axis=0) == 0)[0]
    print(f"  Number of constant voxels: {len(constant_voxels)}")
    if len(constant_voxels) > 0:
        print(f"  Indices of some constant voxels: {constant_voxels[:10]}...")

    correlations = []
    for i, voxel_ts in enumerate(brain_time_series.T):
        if np.std(voxel_ts) == 0:
            correlations.append(np.nan)
        else:
            corr = np.corrcoef(phys, voxel_ts)[0,1]
            correlations.append(corr)

        if i % 100000 == 0:
            print(f"  Processed {i} voxels")

    correlations = np.array(correlations)
    print(f"  Number of NaN correlations: {np.isnan(correlations).sum()}")
    return np.arctanh(correlations)

def visualize_roi_sphere(coords, radius, output_file):
    mni_template = datasets.load_mni152_template()
    roi_sphere = maskers.NiftiSpheresMasker([tuple(coords)], radius=radius).fit()
    roi_img = roi_sphere.mask_img_
    
    display = plotting.plot_roi(roi_img, bg_img=mni_template, 
                                cut_coords=coords, display_mode='ortho',
                                title="ROI Sphere Visualization")
    #display.savefig(output_file)
    #display.close()

def conduct_analyses():
    start_time = time.time()
    for ss in subs:
        subject_start_time = time.time()
        print(f"Processing subject: {ss}")
        sub_dir = f'{study_dir}/{ss}/ses-01/'
        roi_dir = f'{sub_dir}/derivatives/rois'
        exp = 'loc'
        exp_dir = f'{sub_dir}/derivatives/fsl/{exp}'
        out_dir = os.path.join(sub_dir, 'derivatives', 'fc')
        os.makedirs(out_dir, exist_ok=True)

        roi_coords = pd.read_csv(f'{roi_dir}/spheres/sphere_coords.csv')
        print(roi_coords.head())

        for rr in rois:
            all_ppi_runs = []
            all_fc_runs = []
            
            for rcn, rc in enumerate(run_combos):
                curr_coords = roi_coords[(roi_coords['index'] == rcn) & 
                                        (roi_coords['task'] == exp) & 
                                        (roi_coords['roi'] == rr)]
                
                if curr_coords.empty:
                    print(f"No coordinates found for {ss}, {rr}, {exp}, RC: {rc}")
                    continue

                peak_coords = curr_coords[['x', 'y', 'z']].values[0]

                held_out_run = list(set(range(1, 4)) - set(rc))[0]
                img4d = image.load_img(f'{exp_dir}/run-0{held_out_run}/1stLevel.feat/filtered_func_data_reg.nii.gz')
                
                print(f"Original img4d shape: {img4d.shape}")
                print(f"Any NaNs in original img4d: {np.isnan(img4d.get_fdata()).any()}")
                
                img4d = image.clean_img(img4d, standardize=True)
                print(f"Cleaned img4d shape: {img4d.shape}")
                print(f"Any NaNs in cleaned img4d: {np.isnan(img4d.get_fdata()).any()}")

                phys = extract_roi_sphere(img4d, peak_coords)
                print(f"Phys shape: {phys.shape}")
                print(f"Any NaNs in phys: {np.isnan(phys).any()}")

                psy = make_psy_cov([held_out_run], ss)
                print(f"Psy shape: {psy.shape}")
                print(f"Any NaNs in psy: {np.isnan(psy).any()}")

                brain_time_series = brain_masker.fit_transform(img4d)
                print(f"Brain time series shape: {brain_time_series.shape}")
                print(f"Any NaNs in brain time series: {np.isnan(brain_time_series).any()}")
                
                fc_correlations = calculate_fc(brain_time_series, phys)
                fc_img = brain_masker.inverse_transform(fc_correlations)
                all_fc_runs.append(fc_img)

                confounds = pd.DataFrame({'psy': psy[:,0], 'phys': phys[:,0]})
                ppi = psy * phys
                ppi = ppi.reshape((ppi.shape[0], 1))
                brain_time_series = brain_masker.fit_transform(img4d, confounds=confounds)
                seed_to_voxel_correlations = (np.dot(brain_time_series.T, ppi) / ppi.shape[0])
                seed_to_voxel_correlations = np.arctanh(seed_to_voxel_correlations)
                ppi_img = brain_masker.inverse_transform(seed_to_voxel_correlations.T)
                all_ppi_runs.append(ppi_img)

                print(f"{ss}, {rr}, {exp}, RC: {rc}, PPI max: {seed_to_voxel_correlations.max()}")

                # Visualize ROI sphere
                visualize_roi_sphere(peak_coords, radius=6, output_file=os.path.join(out_dir, f'{ss}_{rr}_RC{rc[0]}{rc[1]}_roi_sphere.png'))

            mean_ppi = image.mean_img(all_ppi_runs)
            mean_fc = image.mean_img(all_fc_runs)
            
            nib.save(mean_ppi, os.path.join(out_dir, f'{ss}_{rr}_{exp}_ppi.nii.gz'))
            nib.save(mean_fc, os.path.join(out_dir, f'{ss}_{rr}_{exp}_fc.nii.gz'))
        
        subject_end_time = time.time()
        print(f"Finished processing subject {ss}. Time taken: {(subject_end_time - subject_start_time) / 60:.2f} minutes")
    
    end_time = time.time()
    print(f"Total processing time: {(end_time - start_time) / 3600:.2f} hours")

if __name__ == "__main__":
    conduct_analyses()