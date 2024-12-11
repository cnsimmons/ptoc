import os
import subprocess
import pandas as pd
import sys

sys.path.insert(0, '/user_data/csimmon2/git_repos/ptoc')
import ptoc_params as params

raw_dir = params.raw_dir
sub_info_path = '/user_data/csimmon2/git_repos/ptoc/sub_info_tool.csv'
sub_info = pd.read_csv(sub_info_path)
target_subjects = sub_info[sub_info['exp'] == 'spaceloc']['sub'].tolist()

for sub in target_subjects:
    sub_dir = f'{raw_dir}/{sub}/ses-01'
    anat = f'{raw_dir}/{sub}/ses-01/anat/{sub}_ses-01_T1w_brain.nii.gz'
    
    for run in range(1, 3):
        print(sub, run)
        # Input zstat path matches your previous registration
        zstat_path = f'/user_data/csimmon2/temp_derivatives/{sub}/ses-01/derivatives/stats/zstat3_reg_run{run}.nii.gz'
        # Output path in same directory
        out_path = f'/user_data/csimmon2/temp_derivatives/{sub}/ses-01/derivatives/stats/zstat3_mni_run{run}.nii.gz'
        # Transform matrix from original feat directory
        run_dir = f'{sub_dir}/derivatives/fsl/toolloc/run-0{run}/1stLevel.feat'
        transform_mat = f'{run_dir}/reg/example_func2standard.mat'
        
        if os.path.exists(zstat_path) and os.path.exists(transform_mat):
            print(f"Processing {sub} run {run}")
            cmd = f'flirt -in {zstat_path} -ref {anat} -out {out_path} -applyxfm -init {transform_mat} -interp trilinear'
            subprocess.run(cmd.split(), check=True)
        else:
            print(f"Missing files for {sub} run {run}")