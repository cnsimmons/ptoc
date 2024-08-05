import os
import pandas as pd
import numpy as np
from nilearn import image, input_data, plotting
from nilearn.maskers import NiftiMasker
from nilearn.datasets import load_mni152_brain_mask
from nilearn.glm.first_level import compute_regressor, FirstLevelModel
import nibabel as nib
from multiprocessing import Pool
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
raw_dir = params.raw_dir #'/lab_data/behrmannlab/vlad/hemispace'

#sub_info = pd.read_csv(f'{curr_dir}/sub_info.csv')
#subs = sub_info[sub_info['group'] == 'control']['sub'].tolist()
#rois = ['pIPS', 'LO', 'V1'] 
subs = ['sub-025']
rois = ['V1']

run_num = 3
run_combos = list(itertools.combinations(range(1, run_num + 1), 2)) #enumerate runs

whole_brain_mask = load_mni152_brain_mask()
brain_masker = NiftiMasker(whole_brain_mask, smoothing_fwhm=0, standardize=True)

def extract_roi_sphere(img, coords, radius=6):
    roi_masker = input_data.NiftiSpheresMasker([tuple(coords)], radius=radius)
    seed_time_series = roi_masker.fit_transform(img)
    return np.mean(seed_time_series, axis=1).reshape(-1, 1)

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
    
    # Check if we have any events outside the expected time range
    if full_cov['onset'].max() >= times[-1] or full_cov['onset'].min() < 0:
        raise ValueError("Event onsets are outside the expected time range")
    
    cov = full_cov.to_numpy()
    
    psy, _ = compute_regressor(cov.T, 'spm', times)
    
    # Ensure psy has exactly vols * len(runs) timepoints
    if len(psy) != vols * len(runs):
        raise ValueError(f"Generated psy does not match expected length. Expected {vols * len(runs)}, got {len(psy)}")
    
    return psy

def calculate_fc(brain_time_series, phys):
    min_length = min(brain_time_series.shape[0], phys.shape[0])
    brain_time_series = brain_time_series[:min_length]
    phys = phys[:min_length]
    
    correlations = np.array([np.corrcoef(phys.ravel(), voxel_ts)[0,1] 
                             for voxel_ts in brain_time_series.T])
    
    return np.arctanh(correlations)

def find_peak_voxel(roi_parcel, localizer_data):
    combined_activation = image.math_img('np.sum(np.abs(np.array(img)), axis=3)',
                                         img=localizer_data)
    masked_activation = image.math_img('img1 * (img2 > 0)',
                                       img1=combined_activation,
                                       img2=roi_parcel)
    data = masked_activation.get_fdata()
    peak_idx = np.unravel_index(np.argmax(data), data.shape)
    peak_coords = nib.affines.apply_affine(masked_activation.affine, peak_idx)
    return peak_coords

def visualize_peak_voxel(anatomical_img, roi_parcel, peak_coords, output_file):
    display = plotting.plot_anat(anatomical_img, cut_coords=peak_coords)
    display.add_contours(roi_parcel, levels=[0.5], colors='r')
    display.add_markers(peak_coords, marker_color='g', marker_size=100)
    display.savefig(output_file)
    display.close()

def check_roi_registration(roi_img, output_file):
    """
    Check the registration of the ROI to MNI space.
    
    Args:
    roi_img (Nifti1Image): The ROI image.
    output_file (str): Path to save the output image.
    """
    mni_template = load_mni152_template()
    
    display = plotting.plot_roi(roi_img, bg_img=mni_template, 
                                cut_coords=(0, 0, 0), display_mode='ortho',
                                title="ROI Registration Check")
    display.savefig(output_file)
    display.close()

def conduct_analyses():
    start_time = time.time()
    for ss in subs:
        subject_start_time = time.time()
        print(f"Processing subject: {ss}")
        sub_dir = f'{study_dir}/{ss}/ses-01/'
        roi_dir = f'{sub_dir}/derivatives/rois'
        exp = 'loc'
        exp_dir = f'{sub_dir}/derivatives/fsl/{exp}'
        out_dir = os.path.join(sub_dir, 'derivatives', 'fc_ppi')
        os.makedirs(out_dir, exist_ok=True)

        roi_coords = pd.read_csv(f'{roi_dir}/spheres/sphere_coords.csv')

        for rr in rois:
            all_ppi_runs = []
            all_fc_runs = []
            
            # Check ROI registration
            roi_img = nib.load(os.path.join(roi_dir, 'parcels', f'{rr}.nii.gz'))
            check_roi_registration(roi_img, os.path.join(out_dir, f'{ss}_{rr}_registration_check.png'))
            
            for rcn, rc in enumerate(run_combos):
                curr_coords = roi_coords[(roi_coords['index'] == rcn) & 
                                         (roi_coords['task'] == exp) & 
                                         (roi_coords['roi'] == rr)]

                # Load localizer data for peak voxel finding
                localizer_data = [
                    image.load_img(f'{exp_dir}/run-0{rn}/1stLevel.feat/stats/zstat1.nii.gz')
                    for rn in rc
                ]
                
                roi_parcel = nib.load(os.path.join(roi_dir, 'parcels', f'{rr}.nii.gz'))
                peak_coords = find_peak_voxel(roi_parcel, localizer_data)
                
                # Visualize peak voxel
                anat_img = nib.load(os.path.join(hemispace_dir, ss, 'ses-01', 'anat', f'{ss}_ses-01_T1w.nii.gz'))
                visualize_peak_voxel(anat_img, roi_parcel, peak_coords, 
                                     os.path.join(out_dir, f'{ss}_{rr}_rc{rc[0]}{rc[1]}_peak_voxel.png'))

                # Use the held-out run for connectivity analysis
                held_out_run = list(set(range(1, 4)) - set(rc))[0]
                img4d = image.load_img(f'{exp_dir}/run-0{held_out_run}/1stLevel.feat/filtered_func_data_reg.nii.gz')
                img4d = image.clean_img(img4d, standardize=True)

                phys = extract_roi_sphere(img4d, peak_coords)
                psy = make_psy_cov([held_out_run], ss)

                # FC Analysis
                brain_time_series = brain_masker.fit_transform(img4d)
                fc_correlations = calculate_fc(brain_time_series, phys)
                fc_img = brain_masker.inverse_transform(fc_correlations)
                all_fc_runs.append(fc_img)

                # PPI Analysis
                confounds = pd.DataFrame({'psy': psy[:,0], 'phys': phys[:,0]})
                ppi = psy * phys
                ppi = ppi.reshape((ppi.shape[0], 1))
                brain_time_series = brain_masker.fit_transform(img4d, confounds=confounds)
                seed_to_voxel_correlations = (np.dot(brain_time_series.T, ppi) / ppi.shape[0])
                seed_to_voxel_correlations = np.arctanh(seed_to_voxel_correlations)
                ppi_img = brain_masker.inverse_transform(seed_to_voxel_correlations.T)
                all_ppi_runs.append(ppi_img)

                print(f"{ss}, {rr}, {exp}, RC: {rc}, PPI max: {seed_to_voxel_correlations.max()}")

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