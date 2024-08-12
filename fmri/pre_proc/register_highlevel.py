"""
Register each HighLevel to anat in a parallelized manner

"""
curr_dir = f'/user_data/csimmon2/git_repos/ptoc'
import numpy as np
import pandas as pd
import subprocess
import os
import pdb
from nilearn.datasets import load_mni152_brain_mask, load_mni152_template
import sys
sys.path.append(curr_dir)

import ptoc_params as params

data_dir = params.data_dir
raw_dir = params.raw_dir
results_dir = params.results_dir

sub_info = params.sub_info
task_info = params.task_info

suf = params.suf
thresh = params.thresh
rois = params.rois

runs = params.runs

cope = 5 #3 #5 #short cut until I can recode scripts for task_info csv instead of the discrete
task = 'loc'


mni='/opt/fsl/6.0.3/data/standard/MNI152_T1_2mm_brain.nii.gz' #this is the MNI we use for analysis

for sub in sub_info['sub']:
    sub_dir = f'{data_dir}/{sub}/ses-01'
    anat_dir = f'{raw_dir}/{sub}/ses-01'
    print(f'Registering {sub} {task} to anat')
    #register each highlevel to anat
    zstat = f'{sub_dir}/derivatives/fsl/{task}/HighLevel.gfeat/cope{cope}.feat/stats/zstat1.nii.gz'

    out_func = f'{sub_dir}/derivatives/fsl/{task}/HighLevel.gfeat/cope{cope}.feat/stats/zstat1_reg.nii.gz'

    #check if zstat exists
    if os.path.exists(zstat):
        bash_cmd = f'flirt -in {zstat} -ref {mni} -out {out_func} -applyxfm -init {anat_dir}/anat/anat2stand.mat -interp trilinear'
        subprocess.run(bash_cmd.split(), check=True)
    else:
        print(f'zstat {zstat} does not exist for subject {sub}')