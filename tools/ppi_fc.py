#fc_ppi native space with hemispheres

# runs 2nd level analyses for PPI and FC
import sys
sys.path.insert(0, '/user_data/csimmon2/git_repos/ptoc')
import glob
import pandas as pd
import gc
from nilearn import image, input_data
import numpy as np
import nibabel as nib
import os
from nilearn.glm.first_level import compute_regressor
import warnings
import ptoc_params as params
import argparse

raw_dir = params.raw_dir
results_dir = params.results_dir
warnings.filterwarnings('ignore')

tr = 1
vols = 341

# Explicitly set the path to sub_info.csv
sub_info_path = '/user_data/csimmon2/git_repos/ptoc/sub_info_tool.csv'  # Adjust this path as needed

# Define subjects and ROIs
sub_info = pd.read_csv(sub_info_path)
subs = sub_info[sub_info['exp'] == 'spaceloc']['sub'].tolist()
rois = ['pIPS', 'LO']
hemispheres = ['left', 'right']

'''run info'''
run_num = 2
runs = list(range(1, run_num + 1))
run_combos = []
#determine the number of left out run combos
for rn1 in range(1,run_num+1):
    for rn2 in range(rn1+1,run_num+1):
        run_combos.append([rn1,rn2])

'''''

def extract_roi_coords():
    """
    Define ROIs
    """
    parcels = ['PPC', 'APC']

    for ss in subs:
        sub_dir = f'{study_dir}/sub-{study}{ss}/ses-01'
        roi_dir = f'{sub_dir}/derivatives/rois'
        os.makedirs(f'{roi_dir}/spheres', exist_ok=True)
        
        '''make roi spheres for spaceloc'''
        
        exp_dir = f'{sub_dir}/derivatives/fsl'
        parcel_dir = f'{roi_dir}/parcels'
        roi_coords = pd.DataFrame(columns = ['index','task','roi','x','y','z'])
        for rcn, rc in enumerate(run_combos): #determine which runs to use for creating ROIs
            roi_runs = [ele for ele in runs if ele not in rc]

            #load each run
            all_runs = []
            for rn in roi_runs:
                curr_run = image.load_img(f'{exp_dir}/{exp}/run-0{rn}/1stLevel_roi.feat/stats/zstat1_reg.nii.gz')
        
                all_runs.append(curr_run)

            mean_zstat = image.mean_img(all_runs)
            affine = mean_zstat.affine
            
            #loop through parcel determine coord of peak voxel
            for lr in ['l','r']:
                for pr in parcels:

                    #load parcel
                    roi = image.load_img(f'{parcel_dir}/{lr}{pr}.nii.gz')
                    roi = image.math_img('img > 0', img=roi)

                    #masked_image = roi*image.get_data(mean_zstat)
                    coords = plotting.find_xyz_cut_coords(mean_zstat,mask_img=roi, activation_threshold = .99)

                    masked_stat = image.math_img('img1 * img2', img1=roi, img2=mean_zstat)
                    masked_stat = image.get_data(masked_stat)
                    np_coords = np.where(masked_stat == np.max(masked_stat))
                    #pdb.set_trace()
                    #max_coord = image.coord_transform(np_coords,affine)



                    #masked_image = nib.Nifti1Image(masked_image, affine)  # create the volume image
                    curr_coords = pd.Series([rcn, exp, f'{lr}{pr}'] + coords, index=roi_coords.columns)
                    roi_coords = roi_coords.append(curr_coords,ignore_index = True)

                    '''Create spheres for control task ROIs'''
                    for ct in control_tasks:
                        control_zstat = image.load_img(f'{exp_dir}/{ct}/HighLevel_roi.gfeat/cope1.feat/stats/zstat1.nii.gz')
                        coords = plotting.find_xyz_cut_coords(control_zstat,mask_img=roi, activation_threshold = .99)

                        
                        curr_coords = pd.Series([rcn, ct, f'{lr}{pr}'] + coords, index=roi_coords.columns)
                        roi_coords = roi_coords.append(curr_coords,ignore_index = True)

        roi_coords.to_csv(f'{roi_dir}/spheres/sphere_coords.csv', index=False)

'''


def extract_roi_sphere(img, coords):
    roi_masker = input_data.NiftiSpheresMasker([tuple(coords)], radius=6)
    seed_time_series = roi_masker.fit_transform(img)
    phys = np.mean(seed_time_series, axis=1).reshape(-1, 1)
    return phys

def make_psy_cov(runs, ss):
    temp_dir = f'{raw_dir}/{ss}/ses-01'
    cov_dir = f'{temp_dir}/covs'
    vols, tr = 184, 2.0
    times = np.arange(0, vols * tr, tr)
    full_cov = pd.DataFrame(columns=['onset', 'duration', 'value'])

    for rn in runs:
        ss_num = ss.split('-')[1].replace('spaceloc', '')
        tool_cov_file = f'{cov_dir}/ToolLoc_spaceloc{ss_num}_run{run}_tool.txt'
        nontool_cov_file = f'{cov_dir}/ToolLoc_spaceloc{ss_num}_run{run}_scramble.txt'

        if os.path.exists(tool_cov_file) and os.path.exists(nontool_cov_file):
            tool_cov = pd.read_csv(tool_cov_file, sep='\t', header=None, 
                                 names=['onset', 'duration', 'value'])
            nontool_cov = pd.read_csv(nontool_cov_file, sep='\t', header=None, 
                                    names=['onset', 'duration', 'value'])
            nontool_cov['value'] *= -1

            # Adjust onsets for concatenated runs
            run_offset = vols * runs.index(run)
            tool_cov['onset'] += run_offset
            nontool_cov['onset'] += run_offset
            
            full_cov = pd.concat([full_cov, tool_cov, nontool_cov])

    full_cov = full_cov.sort_values(by=['onset'])
    cov = full_cov.to_numpy()
    psy, _ = compute_regressor(cov.T, 'spm', times)
    return psy

def conduct_analyses():
    for ss in subs:
        print(f"Processing subject: {ss}")
        
        temp_dir = f'{raw_dir}/{sub}/ses-01/derivatives/fsl/toolloc'
        roi_dir = f'{raw_dir}/{sub}/ses-01/derivatives/rois'
        out_dir = f'/user_data/csimmon2/temp_derivatives/{sub}/ses-01/derivatives'
        os.makedirs(f'{out_dir}/fc', exist_ok=True)
        
        mask_path = f'{raw_dir}/{sub}/ses-01/anat/{sub}_ses-01_T1w_brain_mask.nii.gz'
        whole_brain_mask = nib.load(mask_path)
        brain_masker = input_data.NiftiMasker(whole_brain_mask, standardize=True)
            
        roi_coords = pd.read_csv(f'{roi_dir}/spheres/sphere_coords_hemisphere_tools.csv') #need to create this file
        
        out_dir = f'/user_data/csimmon2/temp_derivatives/{sub}/ses-01/derivatives'
        os.makedirs(f'{out_dir}/fc', exist_ok=True)
        
        # subject-specific brain mask
        def get_subject_mask(ss):
            mask_path  = f'{raw_dir}/{ss}/ses-01/anat/{ss}_ses-01_T1w_brain_mask.nii.gz'
            return nib.load(mask_path)
        
        # Load subject-specific mask
        whole_brain_mask = get_subject_mask(ss)
        brain_masker = NiftiMasker(whole_brain_mask, smoothing_fwhm=0, standardize=True)

        for tsk in ['ToolLoc']:
            for rr in rois:
                for hemi in hemispheres:
                    roi_start_time = time.time()
                    print(f"Processing ROI: {rr}, Hemisphere: {hemi}")
                    
                    fc_file = f'{out_dir}/fc/{ss}_{rr}_{hemi}_{tsk}_fc_tool.nii.gz' # object
                    ppi_file = f'{out_dir}/fc/{ss}_{rr}_{hemi}_{tsk}_ppi_tool-scramble.nii.gz'
                    
                    do_fc = not os.path.exists(fc_file)
                    do_ppi = not os.path.exists(ppi_file)
                    do_ppi = True
                    
                    if not do_fc and not do_ppi:
                        print(f'Both FC and PPI files for {rr} {hemi} already exist. Skipping...')
                        continue
                    
                    all_runs_fc = []
                    all_runs_ppi = []
                    
                    for rcn, rc in enumerate(run_combos):
                        combo_start_time = time.time()
                        print(f"Completed run combination {rc} for {rr} {hemi} in {time.time() - combo_start_time:.2f} seconds")
                        curr_coords = roi_coords[(roi_coords['index'] == rcn) & 
                                                 (roi_coords['task'] == tsk) & 
                                                 (roi_coords['roi'] == rr) &
                                                 (roi_coords['hemisphere'] == hemi)]
                        
                        if curr_coords.empty:
                            print(f"No coordinates found for {rr}, {hemi}, run combo {rc}")
                            continue
                        
                        coords = curr_coords[['x', 'y', 'z']].values.tolist()[0]
                        
                        filtered_list = [image.clean_img(image.load_img(f'{temp_dir}/run-0{rn}/1stLevel.feat/filtered_func_data_reg.nii.gz'), standardize=True) for rn in rc]
                        img4d = image.concat_imgs(filtered_list)
                        
                        phys = extract_roi_sphere(img4d, coords)
                        
                        # Ensure phys length matches the number of volumes
                        if phys.shape[0] > 184 * len(rc):
                            phys = phys[:184 * len(rc)]
                        
                        brain_time_series = brain_masker.fit_transform(img4d)
                        
                        if do_fc:
                            # FC Analysis
                            correlations = np.dot(brain_time_series.T, phys) / phys.shape[0]
                            correlations = np.arctanh(correlations.ravel())
                            correlation_img = brain_masker.inverse_transform(correlations)
                            all_runs_fc.append(correlation_img)
                        
                        if do_ppi:
                            # PPI Analysis
                            psy = make_psy_cov(rc, ss)  # Generate psy for the current run combination
                            
                            # Ensure psy length matches phys
                            if psy.shape[0] > phys.shape[0]:
                                psy = psy[:phys.shape[0]]
                            elif psy.shape[0] < phys.shape[0]:
                                phys = phys[:psy.shape[0]]
                                brain_time_series = brain_time_series[:psy.shape[0]]
                            
                            ppi_regressor = phys * psy
                            ppi_correlations = np.dot(brain_time_series.T, ppi_regressor) / ppi_regressor.shape[0]
                            ppi_correlations = np.arctanh(ppi_correlations.ravel())
                            ppi_img = brain_masker.inverse_transform(ppi_correlations)
                            all_runs_ppi.append(ppi_img)
                        
                        combo_end_time = time.time()
                        print(f"Completed run combination {rc} for {rr} {hemi} in {time.time() - combo_start_time:.2f} seconds")
                    
                    if do_fc:
                        mean_fc = image.mean_img(all_runs_fc)
                        nib.save(mean_fc, fc_file)
                        print(f'Saved FC result for {rr} {hemi}')
                    
                    if do_ppi:
                        mean_ppi = image.mean_img(all_runs_ppi)
                        nib.save(mean_ppi, ppi_file)
                        print(f'Saved PPI result for {rr} {hemi}')
                        
                roi_end_time = time.time()
                print(f"Completed {rr} {hemi} in {roi_end_time - roi_start_time:.2f} seconds")
# Call the function
conduct_analyses()