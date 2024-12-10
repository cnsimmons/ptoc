#run to start
curr_dir = '/user_data/csimmon2/git_repos/ptoc'

import sys
sys.path.insert(0, curr_dir)

import os
import subprocess
import pandas as pd

## run in terminal
# module load fsl-6.0.3
#python analyses/subj2parcel.py

import os
import subprocess

# Set up directories and parameters
study_dir = "/user_data/csimmon2/temp_derivatives"
raw_dir = "/lab_data/behrmannlab/vlad/hemispace"
results_dir = "/user_data/csimmon2/git_repos/ptoc/results/tools"
mni_brain = os.path.join(os.environ['FSLDIR'], "data/standard/MNI152_T1_2mm_brain.nii.gz")

# subjects
sub_info = pd.read_csv(f'{curr_dir}/sub_info_tool.csv')
subs = sub_info[sub_info['exp'] == 'spaceloc']['sub'].tolist()

for sub in subs:
    print(f"Processing subject: {sub}")
    sub_dir = f"{study_dir}/{sub}/ses-01"
    out_dir = f"{sub_dir}/derivatives"
    temp_dir = f'{raw_dir}/{sub}/ses-01/derivatives/fsl/toolloc'
    
    # Use existing transformation matrix from functional preprocessing
    anat2mni_mat = f"{temp_dir}/run-01/1stLevel.feat/reg/example_func2standard.mat"

    if not os.path.isfile(anat2mni_mat):
        print(f"Transform matrix not found at {anat2mni_mat} for {sub}. Skipping...")
        continue

    # Loop through ROIs and hemispheres
    rois = ['pIPS', 'LO', 'PFS', 'aIPS']
    hemispheres = ['left', 'right']
    
    for roi in rois:
        for hemi in hemispheres:
            # Match the file naming from your PPI/FC script
            fc_native = f"{out_dir}/fc/{sub}_{roi}_{hemi}_ToolLoc_fc.nii.gz"
            fc_mni = f"{out_dir}/fc/{sub}_{roi}_{hemi}_ToolLoc_fc_mni.nii.gz"
            
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

            # Match the PPI file naming from your script
            ppi_native = f"{out_dir}/fc/{sub}_{roi}_{hemi}_ToolLoc_ppi.nii.gz"
            ppi_mni = f"{out_dir}/fc/{sub}_{roi}_{hemi}_ToolLoc_ppi_mni.nii.gz"
            
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