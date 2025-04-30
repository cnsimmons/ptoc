import os
import sys
import subprocess
import pandas as pd

# Set up directories and parameters
curr_dir = '/user_data/csimmon2/git_repos/ptoc'
study_dir = "/lab_data/behrmannlab/vlad/ptoc"
raw_dir = "/lab_data/behrmannlab/vlad/hemispace"
results_dir = "/user_data/csimmon2/git_repos/ptoc/results/tools"
mni_brain = os.path.join(os.environ['FSLDIR'], "data/standard/MNI152_T1_2mm_brain.nii.gz")

# subjects
sub_info = pd.read_csv(f'{curr_dir}/sub_info_tool.csv')
subs = sub_info[sub_info['exp'] == 'spaceloc']['sub'].tolist()

# ROIs and hemispheres
rois = ['pIPS', 'LO']
hemispheres = ['left', 'right']

# Force reprocessing flag set to True
force_reprocess = True

for sub in subs:
    print(f"Processing subject: {sub}")
    sub_dir = f"{study_dir}/{sub}/ses-01"
    out_dir = f"{sub_dir}/derivatives"
    anat_brain = f"{raw_dir}/{sub}/ses-01/anat/{sub}_ses-01_T1w_brain.nii.gz"

    # Check if anatomical image exists
    if not os.path.isfile(anat_brain):
        print(f"Anatomical image not found for {sub}. Skipping...")
        continue
    
    # Create MNI output directories if they don't exist
    mni_ppi_dir = os.path.join(out_dir, 'ppi', 'mni')
    os.makedirs(mni_ppi_dir, exist_ok=True)

    # Generate transformation matrix
    anat2mni_mat = f"{out_dir}/fc/anat2mni.mat"
    
    # Make sure the directory exists
    os.makedirs(os.path.dirname(anat2mni_mat), exist_ok=True)
    
    # Always regenerate the transformation matrix when force_reprocess is True
    if force_reprocess or not os.path.isfile(anat2mni_mat):
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
    else:
        print(f"Using existing transformation matrix for {sub}")
    
    # Process tools vs nontools contrast
    for roi in rois:
        for hemi in hemispheres:
            # PPI to MNI - tools vs nontools contrast
            tools_vs_nontools_ppi_native = f"{out_dir}/ppi/{sub}_{roi}_{hemi}_tools_vs_nontools_ToolLoc_ppi.nii.gz"
            tools_vs_nontools_ppi_mni = f"{mni_ppi_dir}/{sub}_{roi}_{hemi}_tools_vs_nontools_ToolLoc_ppi_mni.nii.gz"
            
            # Process file if it exists in native space and we're forcing reprocessing or the MNI file doesn't exist
            if os.path.isfile(tools_vs_nontools_ppi_native) and (force_reprocess or not os.path.isfile(tools_vs_nontools_ppi_mni)):
                print(f"Registering tools vs nontools PPI for {sub}, ROI {roi}, Hemisphere {hemi} to MNI space")
                subprocess.run([
                    'flirt',
                    '-in', tools_vs_nontools_ppi_native,
                    '-ref', mni_brain,
                    '-out', tools_vs_nontools_ppi_mni,
                    '-applyxfm',
                    '-init', anat2mni_mat,
                    '-interp', 'trilinear'
                ], check=True)
            elif not force_reprocess and os.path.isfile(tools_vs_nontools_ppi_mni):
                print(f"Tools vs nontools PPI file for {sub}, ROI {roi}, Hemisphere {hemi} already registered to MNI space")
            else:
                print(f"Tools vs nontools PPI file not found for {sub}, ROI {roi}, Hemisphere {hemi}")

    print(f"Conversion to MNI space completed for {sub}.")