#run to start
curr_dir = f'/user_data/csimmon2/git_repos/ptoc'

import sys
sys.path.insert(0,curr_dir)

import os
import subprocess
import pandas as pd

## run in terminal
# module load fsl-6.0.3
#python analyses/subj2parcel.py

import os
import subprocess

# Set up directories and parameters
study_dir = "/lab_data/behrmannlab/vlad/ptoc"
raw_dir = "/lab_data/behrmannlab/vlad/hemispace"  # Update this path
results_dir = "/user_data/csimmon2/git_repos/ptoc/results"
mni_brain = os.path.join(os.environ['FSLDIR'], "data/standard/MNI152_T1_2mm_brain.nii.gz")

# subjects
sub_info = pd.read_csv(f'{curr_dir}/sub_info.csv')
subs = sub_info[sub_info['group'] == 'control']['sub'].tolist()
#subs = ['sub-107']

for sub in subs:
    print(f"Processing subject: {sub}")
    sub_dir = f"{study_dir}/{sub}/ses-01"
    out_dir = f"{sub_dir}/derivatives"
    anat_brain = f"{raw_dir}/{sub}/ses-01/anat/{sub}_ses-01_T1w_brain.nii.gz"

    # Check if anatomical image exists
    if not os.path.isfile(anat_brain):
        print(f"Anatomical image not found for {sub}. Exiting...")
        exit(1)

    # Create output directories if they don't exist
    #os.makedirs(f"{out_dir}/fc_mni", exist_ok=True) #object
    os.makedirs(f"{out_dir}/scramble", exist_ok=True) #scrabmle
    
    # Generate transformation matrix if it doesn't exist
    anat2mni_mat = f"{out_dir}/anat2mni.mat"
    if not os.path.isfile(anat2mni_mat):
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

    # Loop through ROIs and hemispheres
    for roi in ['LO','pIPS']:
        for hemi in ['left', 'right']:
            
            # FC to MNI
            fc_native = f"{out_dir}/fc/{sub}_{roi}_{hemi}_loc_fc.nii.gz"
            fc_mni = f"{out_dir}/fc_mni/{sub}_{roi}_{hemi}_loc_fc_mni.nii.gz"
            
            if os.path.isfile(fc_native) and not os.path.isfile(fc_mni):
                print(f"Registering FC for {sub}, ROI {roi}, Hemisphere {hemi} to MNI space")
                subprocess.run([
                    'flirt',
                    '-in', fc_native,
                    '-ref', mni_brain,
                    '-out', fc_mni,
                    '-applyxfm',
                    '-init', anat2mni_mat,
                    '-interp', 'trilinear'
                ], check=True)
            elif os.path.isfile(fc_mni):
                print(f"FC MNI file already exists for {sub}, ROI {roi}, Hemisphere {hemi}")
            else:
                print(f"FC file not found for {sub}, ROI {roi}, Hemisphere {hemi}")

            # PPI to MNI
            ppi_native = f"{out_dir}/fc/{sub}_{roi}_{hemi}_loc_ppi.nii.gz"
            ppi_mni = f"{out_dir}/fc_mni/{sub}_{roi}_{hemi}_loc_ppi_mni.nii.gz"
            if os.path.isfile(ppi_native) and not os.path.isfile(ppi_mni):
                print(f"Registering PPI for {sub}, ROI {roi}, Hemisphere {hemi} to MNI space")
                subprocess.run([
                    'flirt',
                    '-in', ppi_native,
                    '-ref', mni_brain,
                    '-out', ppi_mni,
                    '-applyxfm',
                    '-init', anat2mni_mat,
                    '-interp', 'trilinear'
                ], check=True)
            elif os.path.isfile(ppi_mni):
                print(f"PPI MNI file already exists for {sub}, ROI {roi}, Hemisphere {hemi}")
            else:
                print(f"PPI file not found for {sub}, ROI {roi}, Hemisphere {hemi}")

    print(f"Conversion to MNI space completed for {sub}.")