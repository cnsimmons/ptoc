#run to start
curr_dir = f'/user_data/csimmon2/git_repos/ptoc'
import sys
sys.path.insert(0,curr_dir)
import os
import subprocess
import pandas as pd

# Set up directories and parameters
study_dir = "/lab_data/behrmannlab/vlad/ptoc"
raw_dir = "/lab_data/behrmannlab/vlad/hemispace"
results_dir = "/user_data/csimmon2/git_repos/ptoc/results"
mni_brain = os.path.join(os.environ['FSLDIR'], "data/standard/MNI152_T1_2mm_brain.nii.gz")

# subjects
sub_info = pd.read_csv(f'{curr_dir}/sub_info.csv')
subs = sub_info[sub_info['group'] == 'control']['sub'].tolist()
#subs = ['sub-107']

for ss in subs:  
    print(f"Processing subject: {ss}")
    sub_dir = f"{study_dir}/{ss}/ses-01"
    out_dir = f"{sub_dir}/derivatives"
    anat_brain = f"{raw_dir}/{ss}/ses-01/anat/{ss}_ses-01_T1w_brain.nii.gz"

    # Check if anatomical image exists
    if not os.path.isfile(anat_brain):
        print(f"Anatomical image not found for {ss}. Skipping...")
        continue

    # Create output directory if it doesn't exist
    os.makedirs(f"{out_dir}/scramble", exist_ok=True)

    # Check if transformation matrix exists
    anat2mni_mat = f"{out_dir}/anat2mni.mat"
    if not os.path.isfile(anat2mni_mat):
        print(f"Transformation matrix not found for {ss}. Skipping...")
        continue

    # Loop through ROIs and hemispheres
    for rr in ['LO', 'pIPS']:  # Changed 'roi' to 'rr' to match your naming convention
        for hemi in ['left', 'right']:
            # New file pattern for conversion
            fc_file = f'{out_dir}/fc/{ss}_{rr}_{hemi}_loc_fc_scramble.nii.gz'  # Updated input file pattern
            fc_mni = f"{out_dir}/scramble/{ss}_{rr}_{hemi}_loc_fc_scramble_mni.nii.gz"

            if os.path.isfile(fc_file):
                if not os.path.isfile(fc_mni):
                    print(f"Registering FC for {ss}, ROI {rr}, Hemisphere {hemi} to MNI space")
                    subprocess.run([
                        'flirt',
                        '-in', fc_file,
                        '-ref', mni_brain,
                        '-out', fc_mni,
                        '-applyxfm',
                        '-init', anat2mni_mat,
                        '-interp', 'trilinear'
                    ], check=True)
                else:
                    print(f"FC MNI file already exists for {ss}, ROI {rr}, Hemisphere {hemi}")
            else:
                print(f"FC file not found for {ss}, ROI {rr}, Hemisphere {hemi}")
                print(f"Expected path: {fc_file}")

    print(f"Conversion to MNI space completed for {ss}.")

print("Script execution completed.")