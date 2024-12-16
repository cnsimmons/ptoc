import os
import sys
import logging
import subprocess
import nibabel as nib
import numpy as np
from nilearn import image
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import your parameters
curr_dir = f'/user_data/csimmon2/git_repos/ptoc'
sys.path.insert(0, curr_dir)
import ptoc_params as params

# Set up directories and parameters
study = 'ptoc'
study_dir = f"/lab_data/behrmannlab/vlad/{study}"
raw_dir = "/lab_data/behrmannlab/vlad/hemispace"
results_dir = '/user_data/csimmon2/git_repos/ptoc/results'
mni_brain = os.path.join(os.environ['FSLDIR'], "data/standard/MNI152_T1_2mm_brain.nii.gz")

# Load subject information
sub_info = pd.read_csv(f'{curr_dir}/sub_info_tool.csv')  # Changed to sub_info_tool.csv
#sub_info = sub_info[sub_info['exp'] == 'spaceloc']  # Changed filter to match spaceloc experiment

sub_info = sub_info[
    (sub_info['exp'] == 'spaceloc') & 
    (sub_info['sub'].isin(['sub-spaceloc1001', 'sub-spaceloc1002', 'sub-spaceloc1003', 
                              'sub-spaceloc1004', 'sub-spaceloc1005', 'sub-spaceloc1006',
                              'sub-spaceloc1007', 'sub-spaceloc1008', 'sub-spaceloc1009',
                              'sub-spaceloc1010']))
]

rois = ['pIPS', 'LO']  # Keeping the same ROIs
hemispheres = ['left', 'right']
localizer = 'Tool'  # Changed from 'Object' to 'Tool'
run_combos = ['12', '21']

def combine_and_transform(subject):
    logging.info(f"Processing subject: {subject}")
    sub_dir = f'{study_dir}/{subject}/ses-01/'
    gca_dir = f'{sub_dir}/derivatives/gca'
    anat_brain = f"{raw_dir}/{subject}/ses-01/anat/{subject}_ses-01_T1w_brain.nii.gz"

    # Check if anatomical image exists
    if not os.path.isfile(anat_brain):
        logging.error(f"Anatomical image not found for {subject}. Exiting...")
        return

    # Ensure gca directory exists
    os.makedirs(gca_dir, exist_ok=True)

    # Check for existing transformation matrix
    anat2mni_mat = f"{sub_dir}/derivatives/anat2mni.mat"
    if not os.path.isfile(anat2mni_mat):
        logging.error(f"Transformation matrix not found for {subject}. Exiting...")
        return

    for roi in rois:
        for hemi in hemispheres:
            logging.info(f"Processing {roi} {hemi} for subject {subject}")

            # Combine searchlight results
            all_run_results = []
            for run_combo in run_combos:
                img_path = f'{gca_dir}/searchlight_result_tool_runs{run_combo}_{roi}_{hemi}.nii.gz'  # Changed to tool
                if not os.path.exists(img_path):
                    logging.warning(f"File not found: {img_path}")
                    continue
                img = nib.load(img_path)
                all_run_results.append(img)

            if not all_run_results:
                logging.error(f"No valid images found for {roi} {hemi}")
                continue

            # Compute the mean image across all run combinations
            mean_img = image.mean_img(all_run_results)

            # Save the combined image in native space
            native_output_path = f'{gca_dir}/combined_tool_{roi}_{hemi}_native.nii.gz'  # Changed to tool
            nib.save(mean_img, native_output_path)
            logging.info(f"Saved combined native image: {native_output_path}")

            # Transform to MNI space
            mni_output_path = f'{gca_dir}/combined_tool_{roi}_{hemi}_mni.nii.gz'  # Changed to tool
            logging.info(f"Registering GCA for {subject}, ROI {roi}, Hemisphere {hemi} to MNI space")
            subprocess.run([
                'flirt',
                '-in', native_output_path,
                '-ref', mni_brain,
                '-out', mni_output_path,
                '-applyxfm',
                '-init', anat2mni_mat,
                '-interp', 'trilinear'
            ], check=True)

    logging.info(f"Combination and transformation to MNI space completed for {subject}.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python combine_and_transform_searchlight.py <subject>")
        sys.exit(1)

    subject = sys.argv[1]
    combine_and_transform(subject)