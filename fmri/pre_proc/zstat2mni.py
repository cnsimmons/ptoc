"""
Register each 1stlevel to anat in a parallelized manner
"""
curr_dir = f'/user_data/csimmon2/git_repos/ptoc'
import numpy as np
import pandas as pd
import subprocess
import os
import pdb
from nilearn import datasets

import sys
sys.path.append(curr_dir)
import ptoc_params as params


# Define paths
data_dir = f'/lab_data/behrmannlab/vlad/hemispace' #hemispace data
firstlevel_suf = ''

# Load the MNI152 template
mni_template = datasets.load_mni152_template()
mni_brain = mni_template

sub_info = pd.read_csv(f'{curr_dir}/sub_info.csv')
#subs = sub_info[sub_info['group'] == 'control']['sub'].tolist()
subs = ['sub-025']
task = 'loc'
#runs = [1,2,3]
runs = [1]
cope = 3

for sub in subs:
    for run in runs:
    print(f"Processing {sub}")
        
        # Define paths for this subject
        sub_dir = os.path.join(data_dir, sub, 'ses-01', 'derivatives', 'fsl', 'loc', 'run-01', '1stLevel.feat')
        zstat_file = os.path.join(sub_dir, 'stats', 'zstat3.nii.gz')
        output_file = os.path.join(sub_dir, 'stats', 'zstat3_mni.nii.gz')
        
        # Check if the required files exist
        if not os.path.exists(zstat_file):
            print(f"zstat file not found for {sub}")
            continue
        
        # Define the transformation matrix
        transform_mat = os.path.join(sub_dir, 'reg', 'example_func2standard.mat')
        
        if not os.path.exists(transform_mat):
            print(f"Transformation matrix not found for {sub}")
            continue
        
        # Run FLIRT to transform zstat to MNI space
        cmd = f"flirt -in {zstat_file} -ref {mni_brain} -out {output_file} -applyxfm -init {transform_mat} -interp trilinear"
        
        try:
            subprocess.run(cmd, shell=True, check=True)
            print(f"Successfully transformed zstat to MNI space for {sub}")
        except subprocess.CalledProcessError:
            print(f"Error transforming zstat to MNI space for {sub}")

    print("Processing complete.")







