import numpy as np
import pandas as pd
import subprocess
import os
import sys

#don't forget into terminal
    #conda activate fmri
    # module load fsl-6.0.3
    
curr_dir = f'/user_data/csimmon2/git_repos/ptoc'
sys.path.append(curr_dir)
import ptoc_params as params

# Load sub_info DataFrame
sub_info = params.sub_info

# Sample data for demonstration
data_dir = params.data_dir
results_dir = params.results_dir
suf = params.suf
raw_dir = params.raw_dir
thresh = params.thresh
rois = params.rois
cope = params.cope
task = 'loc'

## Get indices for the specified range | section changed in 2024
#start_idx = sub_info.index[22]  # Index of the 10th row
#end_idx = sub_info.index[27]   # Index of the 23rd row
#target_subjects = sub_info.loc[start_idx:end_idx, 'sub']

#target_subjects = sub_info['sub']
#controls only
target_subjects = sub_info[sub_info['group'] == 'control']['sub']
#target_subjects = 'sub_057'

for sub in target_subjects:
    sub_data = f'{data_dir}/{sub}/ses-01'
    sub_data2 = f'{raw_dir}/{sub}/ses-01'
    raw_dir = params.raw_dir
    anat = f'{raw_dir}/{sub}/ses-01/anat/{sub}_ses-01_T1w_brain.nii.gz'

    runs = params.runs
    firstlevel_suf = ''
    task_info = params.task_info
    #task_dir = task_info[task]
    task = 'loc'

    for run in runs:
        print(sub, run)
        
        run_dir = f'{sub_data2}/derivatives/fsl/{task}/run-0{run}/1stLevel{firstlevel_suf}.feat'
        filtered_func = f'{run_dir}/filtered_func_data.nii.gz'
        out_func = f'{run_dir}/filtered_func_data_reg.nii.gz'
        
        zstat_func = f'{run_dir}/stats/zstat{cope}.nii.gz'
        zstat_out = f'{run_dir}/stats/zstat{cope}_reg.nii.gz'
 
        # Check if run exists
        if os.path.exists(filtered_func):
            # Register filtered func
            bash_cmd = f'flirt -in {filtered_func} -ref {anat} -out {out_func} -applyxfm -init {run_dir}/reg/example_func2standard.mat -interp trilinear'
            subprocess.run(bash_cmd.split(), check=True)
            
            # Register zstat
            bash_cmd = f'flirt -in {zstat_func} -ref {anat} -out {zstat_out} -applyxfm -init {run_dir}/reg/example_func2standard.mat -interp trilinear'
            subprocess.run(bash_cmd.split(), check=True)

        else:
            print(f'run {run} for task {task} does not exist for subject {sub}')
            print(run_dir)
