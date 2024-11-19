
import nibabel as nib
import numpy as np
import os
import subprocess
# Libraries and Paths
import os
import sys
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import datasets, plotting
from nilearn.maskers import NiftiLabelsMasker
from nilearn.connectome import GroupSparseCovarianceCV
import logging

# Paths and parameters
curr_dir = f'/user_data/csimmon2/git_repos/ptoc'
sys.path.insert(0, curr_dir)
import ptoc_params as params

raw_dir = params.raw_dir
results_dir = '/user_data/csimmon2/git_repos/ptoc/results'


sub = 'sub-084'
run = '1'
task = 'loc'

# Setup paths
data_dir = '/lab_data/behrmannlab/vlad/hemispace'
func_file = f'{raw_dir}/{sub}/ses-01/func/{sub}_ses-01_task-{task}_run-0{run}_bold.nii.gz'
out_dir = f'{raw_dir}/{sub}/ses-01/derivatives/fsl/{task}/run-0{run}/1stLevel.feat'
out_func = f'{out_dir}/filtered_func_data_reg.nii.gz'
anat = f'{data_dir}/{sub}/ses-01/anat/{sub}_ses-01_T1w_brain.nii.gz'

print("Checking paths exist:")
print(f"func_file exists: {os.path.exists(func_file)}")
print(f"anat exists: {os.path.exists(anat)}")
print(f"reg matrix exists: {os.path.exists(f'{out_dir}/reg/example_func2standard.mat')}")

# Run registration
cmd = f'flirt -in {func_file} -ref {anat} -out {out_func} -applyxfm -init {out_dir}/reg/example_func2standard.mat -interp trilinear'
print("\nRunning registration command:")
print(cmd)
subprocess.run(cmd.split(), check=True)