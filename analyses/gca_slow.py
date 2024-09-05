import os
import pandas as pd
import numpy as np
from nilearn import image, input_data, plotting
from nilearn.glm import threshold_stats_img
from statsmodels.tsa.stattools import grangercausalitytests
import nibabel as nib
import warnings

warnings.filterwarnings('ignore')

# Set up directories and parameters
study = 'ptoc'
study_dir = f"/lab_data/behrmannlab/vlad/{study}"
results_dir = '/user_data/csimmon2/git_repos/ptoc/results'
subs = ['sub-038']  # Update this list as needed
rois = ['pIPS', 'LO']
hemispheres = ['left', 'right']
run_num = 3
runs = list(range(1, run_num + 1))
run_combos = [[rn1, rn2] for rn1 in range(1, run_num + 1) for rn2 in range(rn1 + 1, run_num + 1)]

def find_peak_coordinates(stat_img, roi_mask, hemisphere):
    # Threshold the statistical image
    thresholded_img, threshold = threshold_stats_img(stat_img, alpha=0.001, height_control='fpr')
    
    # Apply the ROI mask
    masked_img = image.math_img('img1 * img2', img1=thresholded_img, img2=roi_mask)
    
    # Get the data and affine transformation
    data = masked_img.get_fdata()
    affine = masked_img.affine
    
    # Find the peak coordinate
    if hemisphere == 'left':
        peak_idx = np.unravel_index(np.argmax(data[:data.shape[0]//2]), data.shape)
    else:
        peak_idx = np.unravel_index(np.argmax(data[data.shape[0]//2:]), data.shape)
        peak_idx = (peak_idx[0] + data.shape[0]//2, peak_idx[1], peak_idx[2])
    
    # Convert voxel indices to mm coordinates
    peak_coord = nib.affines.apply_affine(affine, peak_idx)
    
    return peak_coord

def extract_roi_sphere(img, coords):
    roi_masker = input_data.NiftiSpheresMasker([tuple(coords)], radius=6)
    seed_time_series = roi_masker.fit_transform(img)
    
    phys = np.mean(seed_time_series, axis=1)
    phys = phys.reshape((phys.shape[0], 1))
    
    return phys

# ... [keep other functions like make_psy_cov and extract_cond_ts as they were] ...

def conduct_gca_analyses():
    for ss in subs:
        print(f"Processing subject: {ss}")
        sub_dir = f'{study_dir}/{ss}/ses-01/'
        
        out_dir = f'{study_dir}/{ss}/ses-01/derivatives'
        os.makedirs(f'{out_dir}/gca', exist_ok=True)
        
        gca_results = []
        
        for tsk in ['loc']:
            # Load ROI masks (you'll need to specify the correct paths)
            pips_mask = nib.load(f'{sub_dir}/masks/pIPS_mask.nii.gz')
            lo_mask = nib.load(f'{sub_dir}/masks/LO_mask.nii.gz')
            
            for hemi in hemispheres:
                for rcn, rc in enumerate(run_combos):
                    try:
                        # Load statistical map for this run combination
                        stat_img = nib.load(f'{sub_dir}/derivatives/fsl/{tsk}/run-0{rc[0]}/1stLevel.feat/stats/zstat1.nii.gz')
                        
                        # Find peak coordinates
                        pips_coords = find_peak_coordinates(stat_img, pips_mask, hemi)
                        lo_coords = find_peak_coordinates(stat_img, lo_mask, hemi)
                        
                        print(f"pIPS coordinates: {pips_coords}")
                        print(f"LO coordinates: {lo_coords}")
                        
                        # Load and clean images (standardize only)
                        filtered_list = []
                        for rn in rc:
                            curr_run = nib.load(f'{sub_dir}/derivatives/fsl/{tsk}/run-0{rn}/1stLevel.feat/filtered_func_data_reg.nii.gz')
                            curr_run = image.clean_img(curr_run, standardize=True)
                            filtered_list.append(curr_run)
                        
                        img4d = image.concat_imgs(filtered_list)
                        
                        pips_ts = extract_roi_sphere(img4d, pips_coords)
                        lo_ts = extract_roi_sphere(img4d, lo_coords)
                        
                        # Diagnostic prints
                        print(f"pIPS time series shape: {pips_ts.shape}")
                        print(f"pIPS time series stats - min: {np.min(pips_ts)}, max: {np.max(pips_ts)}, mean: {np.mean(pips_ts)}, std: {np.std(pips_ts)}")
                        print(f"LO time series shape: {lo_ts.shape}")
                        print(f"LO time series stats - min: {np.min(lo_ts)}, max: {np.max(lo_ts)}, mean: {np.mean(lo_ts)}, std: {np.std(lo_ts)}")
                        
                        # Extract condition-specific time series
                        psy = make_psy_cov(rc, ss, 'Object')
                        if psy is not None:
                            pips_cond_ts = extract_cond_ts(pips_ts, psy)
                            lo_cond_ts = extract_cond_ts(lo_ts, psy)
                            
                            # Diagnostic prints for condition-specific time series
                            print(f"Condition-specific pIPS time series shape: {pips_cond_ts.shape}")
                            print(f"Condition-specific pIPS time series stats - min: {np.min(pips_cond_ts)}, max: {np.max(pips_cond_ts)}, mean: {np.mean(pips_cond_ts)}, std: {np.std(pips_cond_ts)}")
                            print(f"Condition-specific LO time series shape: {lo_cond_ts.shape}")
                            print(f"Condition-specific LO time series stats - min: {np.min(lo_cond_ts)}, max: {np.max(lo_cond_ts)}, mean: {np.mean(lo_cond_ts)}, std: {np.std(lo_cond_ts)}")
                            
                            neural_ts = pd.DataFrame({'pips': pips_cond_ts.flatten(), 'lo': lo_cond_ts.flatten()})
                            
                            gc_res_pips_to_lo = grangercausalitytests(neural_ts[['lo', 'pips']], 1, verbose=False)
                            gc_res_lo_to_pips = grangercausalitytests(neural_ts[['pips', 'lo']], 1, verbose=False)
                            
                            f_diff = gc_res_pips_to_lo[1][0]['ssr_ftest'][0] - gc_res_lo_to_pips[1][0]['ssr_ftest'][0]
                            
                            gca_results.append({
                                'run_combo': rcn,
                                'hemisphere': hemi,
                                'condition': 'Object',
                                'f_diff': f_diff
                            })
                            print(f"GCA completed for {ss}, {hemi}, run combo {rcn}")
                        else:
                            print(f"Skipping GCA for {ss}, {hemi}, run combo {rcn} due to missing covariate data")
                    except Exception as e:
                        print(f"Error processing {ss}, {hemi}, run combo {rcn}: {str(e)}")
                        import traceback
                        traceback.print_exc()
        
        gca_df = pd.DataFrame(gca_results)
        gca_df.to_csv(f'{out_dir}/gca/{ss}_gca_results.csv', index=False)
        print(f'Saved GCA results for {ss}')

if __name__ == "__main__":
    conduct_gca_analyses()