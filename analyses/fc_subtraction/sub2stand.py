# Setup and Imports

import sys
curr_dir = '/user_data/csimmon2/git_repos/ptoc'
sys.path.insert(0, curr_dir)

import numpy as np
from nilearn import image
from scipy import stats
import pandas as pd
import os
from tqdm import tqdm
import warnings
import subprocess

# Define parameters
study_dir = "/lab_data/behrmannlab/vlad/ptoc"
raw_dir = "/lab_data/behrmannlab/vlad/hemispace"
results_dir = "/user_data/csimmon2/git_repos/ptoc/results"

# Load subject info
sub_info = pd.read_csv(f'{curr_dir}/sub_info.csv')
subs = sub_info[sub_info['group'] == 'control']['sub'].tolist()
mni_brain = os.path.join(os.environ['FSLDIR'], "data/standard/MNI152_T1_2mm_brain.nii.gz")


# Define paths for input files
pIPS_path = f"{study_dir}/{{subject_id}}/ses-01/derivatives/fc/{{subject_id}}_pIPS_{{hemi}}_loc_fc.nii.gz"
LO_path = f"{study_dir}/{{subject_id}}/ses-01/derivatives/fc/{{subject_id}}_LO_{{hemi}}_loc_fc.nii.gz"

# Define output directories
group_out_dir = f'{curr_dir}/analyses/fc_subtraction'
os.makedirs(group_out_dir, exist_ok=True)


# Conversion to Standard Space

mni_brain = os.path.join(os.environ['FSLDIR'], "data/standard/MNI152_T1_2mm_brain.nii.gz")

def convert_to_mni(sub, input_file, output_file, anat2mni_mat, mni_brain):
    if not os.path.isfile(input_file):
        print(f"Input file not found: {input_file}")
        return None

    if os.path.isfile(output_file):
        print(f"MNI file already exists: {output_file}")
        return output_file

    print(f"Converting to MNI space for {sub}")
    subprocess.run([
        'flirt',
        '-in', input_file,
        '-ref', mni_brain,
        '-out', output_file,
        '-applyxfm',
        '-init', anat2mni_mat,
        '-interp', 'trilinear'
    ], check=True)

    return output_file

mni_files = {'left': [], 'right': []}

for sub in tqdm(subs, desc="Converting to MNI space"):
    sub_dir = f"{study_dir}/{sub}/ses-01"
    out_dir = f"{sub_dir}/derivatives"
    fc_dir = f"{out_dir}/fc"
    fc_mni_dir = f"{out_dir}/fc_mni"
    os.makedirs(fc_mni_dir, exist_ok=True)

    # Generate transformation matrix if it doesn't exist
    anat2mni_mat = f"{out_dir}/anat2mni.mat"
    if not os.path.isfile(anat2mni_mat):
        anat_brain = f"{raw_dir}/{sub}/ses-01/anat/{sub}_ses-01_T1w_brain.nii.gz"
        if not os.path.isfile(anat_brain):
            print(f"Anatomical image not found for {sub}. Skipping...")
            continue
        print(f"Generating transformation matrix for {sub}")
        subprocess.run([
            'flirt',
            '-in', anat_brain,
            '-ref', mni_brain,
            '-omat', anat2mni_mat,
            '-bins', '256',
            '-cost', 'corratio',
            '-searchrx', '-90', '90',
            '-searchry', '-90', '90',
            '-searchrz', '-90', '90',
            '-dof', '12'
        ], check=True)

    for hemi in ['left', 'right']:
        input_file = f"{fc_dir}/{sub}_LO_minus_pIPS_{hemi}_fc.nii.gz"
        mni_file = f"{fc_mni_dir}/{sub}_LO_minus_pIPS_{hemi}_fc_mni.nii.gz"
        mni_result = convert_to_mni(sub, input_file, mni_file, anat2mni_mat, mni_brain)
        if mni_result:
            mni_files[hemi].append(mni_result)

print("Conversion to MNI space completed for all subjects.")