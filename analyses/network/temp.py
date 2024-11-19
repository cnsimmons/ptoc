"""
Register each 1stlevel to anat in a parallelized manner
"""
curr_dir = f'/user_data/csimmon2/git_repos/ptoc'
import numpy as np
import pandas as pd
import subprocess
import os
import pdb

import sys
sys.path.append(curr_dir)
import ptoc_params as params
import os
import sys

# Paths and parameters
exp = 'hemispace'
sub = 'sub-084' # Subject ID 
run = '1' # Run number 
task = 'loc' # Task name
#run_1stlevel = True
ses = 1

data_dir = params.data_dir 
results_dir = params.results_dir

sub_info = params.sub_info
task_info = params.task_info
raw_dir = params.raw_dir

suf = params.suf
thresh = params.thresh

# Directory setup
study_dir = f'/lab_data/behrmannlab/vlad/{exp}'
sub_dir = f"{study_dir}/{sub}/ses-0{ses}"
task_dir = f'{sub_dir}/derivatives/fsl/{task}'
cope = params.cope
sub_data = f'{raw_dir}/{sub}/ses-01'
suf = ''

# Check input paths
func_file = f'{study_dir}/{sub}/ses-0{ses}/func/{sub}_ses-0{ses}_task-{task}_run-0{run}_bold.nii.gz'
anat = f'{study_dir}/{sub}/ses-0{ses}/anat/{sub}_ses-0{ses}_T1w_brain.nii.gz'

print("Checking paths exist:")
print(f"func_file exists: {os.path.exists(func_file)}")
print(f"anat exists: {os.path.exists(anat)}")

'''
# Run first level analysis
if run_1stlevel:
   job_cmd = f'feat {task_dir}/run-0{run}/1stLevel{suf}.fsf'
   if os.path.exists(f'{task_dir}/run-0{run}/1stLevel{suf}.fsf'):
       print(f"\nRunning first level analysis")
       os.system(job_cmd)
   else:
       print(f"\nERROR: FSF file not found at {task_dir}/run-0{run}/1stLevel{suf}.fsf")
'''

print (sub, run)

run_dir = f'{sub_data}/derivatives/fsl/{task}/run-0{run}/1stLevel++.feat'
filtered_func = f'{run_dir}/filtered_func_data.nii.gz'
out_func = f'{run_dir}/filtered_func_data_reg.nii.gz'

zstat_func = f'{run_dir}/stats/zstat{cope}.nii.gz'
zstat_out = f'{run_dir}/stats/zstat{cope}_reg.nii.gz'

#check if run exists
if os.path.exists(filtered_func):
    #register filtered func
    bash_cmd = f'flirt -in {filtered_func} -ref {anat} -out {out_func} -applyxfm -init {run_dir}/reg/example_func2standard.mat -interp trilinear'
    subprocess.run(bash_cmd.split(), check=True)
    
    #register zstat
    bash_cmd = f'flirt -in {zstat_func} -ref {anat} -out {zstat_out} -applyxfm -init {run_dir}/reg/example_func2standard.mat -interp trilinear'
    subprocess.run(bash_cmd.split(), check=True)

else:
    print(f'run {run} for task {task} does not exist for subject {sub}')
    print(run_dir)
