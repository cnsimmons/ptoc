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

# Import your parameters
curr_dir = f'/user_data/csimmon2/git_repos/ptoc'
sys.path.insert(0, curr_dir)
import ptoc_params as params
# Set up directories and parameters
study = 'ptoc'
study_dir = f"/lab_data/behrmannlab/vlad/{study}"
results_dir = '/user_data/csimmon2/git_repos/ptoc/results'
raw_dir = params.raw_dir

sub_info = pd.read_csv(f'{curr_dir}/sub_info.csv')
subs = sub_info[sub_info['group'] == 'control']['sub'].tolist()
rois = ['pIPS', 'LO', 'V1'] 
run_num = 3
runs = list(range(1, run_num + 1))

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
    times = np.arange(0, vols * tr, tr)
    full_cov = pd.DataFrame(columns=['onset', 'duration', 'value'])

    for rn in runs:
        ss_num = ss.split('-')[1]
        obj_cov_file = f'{cov_dir}/catloc_{ss_num}_run-0{rn}_Object.txt'
        scr_cov_file = f'{cov_dir}/catloc_{ss_num}_run-0{rn}_Scramble.txt'

        if not os.path.exists(obj_cov_file) or not os.path.exists(scr_cov_file):
            print(f'Covariate file not found for run {rn}')
            continue

        obj_cov = pd.read_csv(obj_cov_file, sep='\t', header=None, names=['onset', 'duration', 'value'])
        scr_cov = pd.read_csv(scr_cov_file, sep='\t', header=None, names=['onset', 'duration', 'value'])
        scr_cov['value'] *= -1
        full_cov = pd.concat([full_cov, obj_cov, scr_cov])

    full_cov = full_cov.sort_values(by=['onset']).reset_index(drop=True)
    cov = full_cov.to_numpy()
    valid_onsets = cov[:, 0] < times[-1]
    cov = cov[valid_onsets]

    if cov.shape[0] == 0:
        print('No valid covariate data after filtering. Returning zeros array.')
        return np.zeros((vols, 1))

    psy, _ = compute_regressor(cov.T, 'spm', times)
    #return psy
    pass

def calculate_fc(brain_time_series, phys):
    min_length = min(brain_time_series.shape[0], phys.shape[0])
    brain_time_series = brain_time_series[:min_length]
    phys = phys[:min_length]
    
    correlations = np.array([np.corrcoef(phys.ravel(), voxel_ts)[0,1] 
                             for voxel_ts in brain_time_series.T])
    
    return np.arctanh(correlations)

def conduct_ppi_analysis(img4d, phys, psy, brain_masker):
    min_length = min(phys.shape[0], psy.shape[0])
    phys = phys[:min_length]
    psy = psy[:min_length]
    
    design_matrix = pd.DataFrame({
        'psy': psy.ravel(),
        'phys': phys.ravel(),
        'interaction': psy.ravel() * phys.ravel()
    })
    
    brain_time_series = brain_masker.fit_transform(img4d)
    
    fmri_glm = FirstLevelModel(t_r=2.0, noise_model='ar1')
    fmri_glm = fmri_glm.fit(brain_time_series, design_matrix=design_matrix)
    
    ppi_contrast = fmri_glm.compute_contrast('interaction')
    
    return ppi_contrast.get_fdata()

def check_registration(stat_map, output_file):
    display = plotting.plot_stat_map(stat_map, threshold=0.01, cut_coords=(0, 0, 0),
                                     display_mode='ortho', black_bg=False)
    display.savefig(output_file)
    display.close()

def visualize_peak_voxel(roi_parcel, peak_coords, output_file):
    display = plotting.plot_roi(roi_parcel, cut_coords=peak_coords, 
                                display_mode='ortho', black_bg=False)
    display.add_markers(peak_coords, marker_color='r', marker_size=100)
    display.savefig(output_file)
    display.close()

def find_peak_voxel(roi_parcel, localizer_data):
    # Implement peak voxel finding logic here
    # This is a placeholder implementation
    roi_mask = roi_parcel.get_fdata() > 0
    combined_data = np.mean([d.get_fdata() for d in localizer_data], axis=0)
    masked_data = combined_data * roi_mask
    peak_index = np.unravel_index(np.argmax(masked_data), masked_data.shape)
    return roi_parcel.affine[:3, :3].dot(peak_index) + roi_parcel.affine[:3, 3]

def load_and_preprocess(file_path):
    img = image.load_img(file_path)
    return image.clean_img(img, standardize=True)

def process_roi(args):
    ss, rr, tsk, sub_dir, temp_dir, out_dir = args
    
    print(f"Processing subject: {ss}, ROI: {rr}, Task: {tsk}")
    
    roi_dir = f'{sub_dir}derivatives/rois'
    
    # Load localizer data for all runs
    localizer_data = [load_and_preprocess(f'{temp_dir}/run-0{rn}/1stLevel.feat/filtered_func_data_reg.nii.gz') for rn in range(1, 4)]
    
    # Find peak voxel using runs 1 and 2
    roi_parcel = nib.load(f'{roi_dir}/{rr}_parcel.nii.gz')
    peak_coords = find_peak_voxel(roi_parcel, localizer_data[:2])
    
    # Visualize peak voxel
    visualize_peak_voxel(roi_parcel, peak_coords, f'{out_dir}/{ss}_{rr}_peak_voxel.png')
    
    # Use run 3 for connectivity analysis
    img4d = localizer_data[2]
    
    # Extract time series from sphere around peak voxel
    phys = extract_roi_sphere(img4d, peak_coords)
    
    brain_time_series = brain_masker.fit_transform(img4d)
    
    # FC Analysis
    fc_correlations = calculate_fc(brain_time_series, phys)
    fc_img = brain_masker.inverse_transform(fc_correlations)
    
    # PPI Analysis
    psy = make_psy_cov([3], ss)  # Use only run 3
    ppi_img = brain_masker.inverse_transform(conduct_ppi_analysis(img4d, phys, psy, brain_masker))
    
    # Check registration
    check_registration(fc_img, f'{out_dir}/{ss}_{rr}_{tsk}_fc_reg_check.png')
    check_registration(ppi_img, f'{out_dir}/{ss}_{rr}_{tsk}_ppi_reg_check.png')
    
    return fc_img, ppi_img

def conduct_analyses():
    start_time = time.time()
    for ss in subs:
        subject_start_time = time.time()
        print(f"Starting processing for subject: {ss}")
        sub_dir = f'{study_dir}/{ss}/ses-01/'
        temp_dir = f'{raw_dir}/{ss}/ses-01/derivatives/fsl/loc'
        
        out_dir = f'{study_dir}/{ss}/ses-01/derivatives/fc'
        os.makedirs(out_dir, exist_ok=True)

        args_list = []
        for tsk in ['loc']:
            for rr in rois:
                args_list.append((ss, rr, tsk, sub_dir, temp_dir, out_dir))

        with Pool(processes=8) as pool:  # Adjust the number of processes as needed
            results = pool.map(process_roi, args_list)

        # Save results
        for (ss, rr, tsk, *_), (fc_img, ppi_img) in zip(args_list, results):
            nib.save(fc_img, f'{out_dir}/{ss}_{rr}_{tsk}_fc_sandbox.nii.gz')
            nib.save(ppi_img, f'{out_dir}/{ss}_{rr}_{tsk}_ppi_sandbox.nii.gz')
            print(f'Saved FC and PPI results for {ss}, {rr}')
            
            subject_end_time = time.time()
        print(f"Finished processing subject {ss}. Time taken: {(subject_end_time - subject_start_time) / 60:.2f} minutes")
    
    end_time = time.time()
    print(f"Total processing time: {(end_time - start_time) / 3600:.2f} hours")

if __name__ == "__main__":
    conduct_analyses()