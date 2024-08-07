"""
Register each 1stLevel to MNI (anat) in a parallelized manner

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

data_dir = f'/lab_data/behrmannlab/vlad/hemispace' #hemispace data
sub_info = pd.read_csv(f'{curr_dir}/sub_info.csv')
subs = sub_info[sub_info['group'] == 'control']['sub'].tolist()
task = 'loc'
runs = [1,2,3]
cope = 3

mni='/opt/fsl/6.0.3/data/standard/MNI152_T1_2mm_brain.nii.gz' #this is the MNI we use for analysis

for sub in subs:
    sub_dir = f'{data_dir}/{sub}/ses-01'
    for run in runs:
        print(f'Registering {sub} run-0{run} to anat and then to MNI')
        
        #register each FirstLevel to anat
        zstat = f'{sub_dir}/derivatives/fsl/loc/run-0{run}/1stLevel.feat/stats/zstat{cope}.nii.gz'
        
        # Output paths
        out_func_anat = f'{sub_dir}/derivatives/fsl/loc/run-0{run}/1stLevel.feat/stats/zstat{cope}_anat.nii.gz'
        out_func_mni = f'{sub_dir}/derivatives/fsl/loc/run-0{run}/1stLevel.feat/stats/zstat{cope}_mni.nii.gz'
        
        # Reference anatomical image
        anat_ref = f'{sub_dir}/anat/{sub}_ses-01_T1w_brain.nii.gz'  # Adjust if needed, maybe T1w.nii.gz

        if os.path.exists(zstat):
            # Step 1: Register first-level to anatomical
            bash_cmd1 = f'flirt -in {zstat} -ref {anat_ref} -out {out_func_anat} -applyxfm -init {sub_dir}/derivatives/fsl/loc/run-0{run}/1stLevel.feat/reg/example_func2highres.mat -interp trilinear'
            subprocess.run(bash_cmd1.split(), check=True)
            
            # Step 2: Register anatomical to MNI
            bash_cmd2 = f'flirt -in {out_func_anat} -ref {mni} -out {out_func_mni} -applyxfm -init {sub_dir}/anat/anat2stand.mat -interp trilinear'
            subprocess.run(bash_cmd2.split(), check=True)
            
            print(f'Completed registration for {sub} run-0{run}')
        else:
            print(f'zstat {zstat} does not exist for subject {sub} run-0{run}')