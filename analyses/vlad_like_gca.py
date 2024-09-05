import os
import pandas as pd
import numpy as np
from nilearn import image, input_data, glm, plotting
from statsmodels.tsa.stattools import grangercausalitytests
import nibabel as nib
import warnings

warnings.filterwarnings('ignore')

# Set up directories and parameters
study = 'ptoc'
study_dir = f"/lab_data/behrmannlab/vlad/{study}"
results_dir = '/user_data/csimmon2/git_repos/ptoc/results'
raw_dir = f"/lab_data/behrmannlab/vlad/hemispace"
subs = ['sub-025']  # Update this list as needed
d_rois = ['pIPS']
v_rois = ['LO']
hemispheres = ['l', 'r']
run_num = 3
runs = list(range(1, run_num + 1))
run_combos = [[rn1, rn2] for rn1 in range(1, run_num + 1) for rn2 in range(rn1 + 1, run_num + 1)]

def extract_roi_sphere(img, coords):
    roi_masker = input_data.NiftiSpheresMasker([tuple(coords)], radius=6)
    seed_time_series = roi_masker.fit_transform(img)
    
    phys = np.mean(seed_time_series, axis=1)
    phys = phys.reshape((phys.shape[0], 1))
    
    return phys

def make_psy_cov(runs, ss, cond='Object'):
    sub_dir = f'{raw_dir}/{ss}/ses-01'
    cov_dir = f'{sub_dir}/covs'
    vols, tr = 184, 2.0
    times = np.arange(0, vols * len(runs) * tr, tr)
    
    full_cov = pd.DataFrame(columns=['onset', 'duration', 'value'])
    
    for rn in runs:
        ss_num = ss.split('-')[1]
        cov_file = f'{cov_dir}/catloc_{ss_num}_run-0{rn}_{cond}.txt'
        
        if not os.path.exists(cov_file):
            print(f'Covariate file not found for run {rn}')
            continue
        
        curr_cov = pd.read_csv(cov_file, sep='\t', header=None, names=['onset', 'duration', 'value'])
        curr_cov['onset'] += (rn - 1) * vols * tr
        full_cov = full_cov.append(curr_cov)
    
    full_cov = full_cov.sort_values(by=['onset'])
    cov = full_cov.to_numpy()
    cov = cov.astype(float)
    
    psy, _ = glm.first_level.compute_regressor(cov.T, 'spm', times)
    psy[psy > 0] = 1
    psy[psy <= 0] = 0
    
    return psy

def extract_cond_ts(ts, cov):
    block_ind = (cov == 1)
    block_ind = np.insert(block_ind, 0, True)
    block_ind = np.delete(block_ind, len(block_ind) - 1)
    block_ind = (cov == 1).reshape((len(cov))) | block_ind
    
    new_ts = ts[block_ind]
    
    return new_ts

def get_roi_coords(zstat_img, roi_mask):
    coords = plotting.find_xyz_cut_coords(zstat_img, mask_img=roi_mask, activation_threshold=0.99)
    return coords

def conduct_gca():
    print('Running GCA...')
    
    for ss in subs:
        sub_summary = pd.DataFrame(columns=['sub', 'fold', 'condition', 'origin', 'target', 'f_diff'])
        
        sub_dir = f'{study_dir}/{ss}/ses-01/'
        roi_dir = f'{sub_dir}derivatives/rois'
        temp_dir = f'{raw_dir}/{ss}/ses-01/derivatives/fsl/loc'
        
        out_dir = f'{study_dir}/{ss}/ses-01/derivatives'
        os.makedirs(f'{out_dir}/results/beta_ts', exist_ok=True)
        
        for rcn, rc in enumerate(run_combos):
            filtered_list = [image.clean_img(nib.load(f'{temp_dir}/run-0{rn}/1stLevel.feat/filtered_func_data_reg.nii.gz'), standardize=True) for rn in rc]
            img4d = image.concat_imgs(filtered_list)
            
            psy = make_psy_cov(rc, ss)
            
            for hemi in hemispheres:
                # Load zstat image (adjust the path as needed)
                zstat_img = nib.load(f'{temp_dir}/HighLevel_roi.gfeat/cope3.feat/stats/zstat1.nii.gz')
                
                for drr in d_rois:
                    # Load dorsal ROI mask
                    dorsal_mask = nib.load(f'{roi_dir}/parcels/{hemi}{drr}.nii.gz')
                    dorsal_coords = get_roi_coords(zstat_img, dorsal_mask)
                    dorsal_ts = extract_roi_sphere(img4d, dorsal_coords)
                    dorsal_cond = extract_cond_ts(dorsal_ts, psy)
                    
                    for vrr in v_rois:
                        # Load ventral ROI mask
                        ventral_mask = nib.load(f'{roi_dir}/parcels/{hemi}{vrr}.nii.gz')
                        ventral_coords = get_roi_coords(zstat_img, ventral_mask)
                        ventral_ts = extract_roi_sphere(img4d, ventral_coords)
                        ventral_cond = extract_cond_ts(ventral_ts, psy)
                        
                        neural_ts = pd.DataFrame({'dorsal': np.squeeze(dorsal_cond), 'ventral': np.squeeze(ventral_cond)})
                        
                        gc_res_dorsal = grangercausalitytests(neural_ts[['ventral', 'dorsal']], 1, verbose=False)
                        gc_res_ventral = grangercausalitytests(neural_ts[['dorsal', 'ventral']], 1, verbose=False)
                        
                        f_diff = gc_res_dorsal[1][0]['ssr_ftest'][0] - gc_res_ventral[1][0]['ssr_ftest'][0]
                        
                        curr_data = pd.Series([ss, rcn, 'Object', f'{hemi}{drr}', f'{hemi}{vrr}', f_diff], index=sub_summary.columns)
                        sub_summary = sub_summary.append(curr_data, ignore_index=True)
                        
                        print(f"Processed {ss}, {hemi}{drr}-{hemi}{vrr}, run combo {rcn}")
        
        print(f'Done GCA for {ss}')
        sub_summary.to_csv(f'{out_dir}/results/beta_ts/gca_summary.csv', index=False)

def summarize_gca():
    print('Creating summary across subjects...')
    
    df_summary = pd.DataFrame()
    
    for ss in subs:
        sub_dir = f'{study_dir}/{ss}/ses-01/'
        data_dir = f'{sub_dir}/derivatives/results/beta_ts'
        
        curr_df = pd.read_csv(f'{data_dir}/gca_summary.csv')
        curr_df = curr_df.groupby(['condition', 'origin', 'target']).mean()
        
        curr_data = [ss]
        col_index = ['sub']
        
        for hemi in hemispheres:
            for drr in d_rois:
                for vrr in v_rois:
                    col_name = f'Object_{hemi}{drr}_{hemi}{vrr}'
                    col_index.append(col_name)
                    curr_data.append(curr_df['f_diff']['Object', f'{hemi}{drr}', f'{hemi}{vrr}'])
        
        if ss == subs[0]:
            df_summary = pd.DataFrame(columns=col_index)
        
        df_summary = df_summary.append(pd.Series(curr_data, index=col_index), ignore_index=True)
    
    df_summary.to_csv(f"{results_dir}/gca/all_roi_summary.csv", index=False)

if __name__ == "__main__":
    conduct_gca()
    summarize_gca()
    print("GCA analysis completed.")