#conda activate fmri
#module load fsl-6.0.3

import numpy as np
import pandas as pd
import subprocess
import os
import sys

sys.path.insert(0, '/user_data/csimmon2/git_repos/ptoc')
import ptoc_params as params

raw_dir = params.raw_dir
sub_info_path = '/user_data/csimmon2/git_repos/ptoc/sub_info_tool.csv'
sub_info = pd.read_csv(sub_info_path)
target_subjects = sub_info[sub_info['exp'] == 'spaceloc']['sub'].tolist()
zstats = [3, 8] # if I want 8 and 11 they are saved elsewhere due to disk space issues, I ran for 3 and 4, had to delete 4 so I could save 8

for sub in target_subjects:
    sub_dir = f'{raw_dir}/{sub}/ses-01'
    anat = f'{raw_dir}/{sub}/ses-01/anat/{sub}_ses-01_T1w_brain.nii.gz'
    
    for run in range(1, 3):
        print(sub, run)
        run_dir = f'{sub_dir}/derivatives/fsl/toolloc/run-0{run}/1stLevel.feat'
        
        for zstat in zstats:
            zstat_func = f'{run_dir}/stats/zstat{zstat}.nii.gz'
            out_dir = f'/user_data/csimmon2/temp_derivatives/{sub}/ses-01/derivatives/stats'
            os.makedirs(out_dir, exist_ok=True)
            zstat_out = f'{out_dir}/zstat{zstat}_reg_run{run}.nii.gz'

            if os.path.exists(zstat_func):
                bash_cmd = f'flirt -in {zstat_func} -ref {anat} -out {zstat_out} -applyxfm -init {run_dir}/reg/example_func2standard.mat -interp trilinear'
                subprocess.run(bash_cmd.split(), check=True)
            else:
                print(f'zstat{zstat}.nii.gz not found for subject {sub}, run {run}')