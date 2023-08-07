"""
Register each 1stlevel to anat in a parallelized manner
#python3 register_1stlevel.py 007
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

sub = sys.argv[1] #which subject

data_dir = params.data_dir #params data dir is from ptoc, but I need hemispace for this script.
results_dir = params.results_dir

sub_info = params.sub_info
task_info = params.task_info
raw_dir = params.raw_dir

suf = params.suf
thresh = params.thresh
rois = params.rois

runs = params.runs
firstlevel_suf = ''

cope = params.cope

sub_data = f'{data_dir}/{sub}/ses-01'


anat = f'{raw_dir}/{sub}/ses-01/anat/{sub}_ses-01_T1w_brain.nii.gz' #brain extracted anat

task = 'loc' #tasks = ['loc']; ses = 1; runs = [1,2,3]


#for task in task_info['task']:
for task in ['loc']:
    for run in runs:
        
        print (sub, run)
        
        run_dir = f'{sub_data}/derivatives/fsl/{task}/run-0{run}/1stLevel{firstlevel_suf}.feat'
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
