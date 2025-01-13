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
sub_info = pd.read_csv(f'{curr_dir}/sub_info_tool.csv')

sub_info = sub_info[(sub_info['exp'] == 'spaceloc')]
    # & 
    #(sub_info['sub'].isin(['sub-spaceloc1001', 'sub-spaceloc1002', 'sub-spaceloc1003', 
                             # 'sub-spaceloc1004', 'sub-spaceloc1005', 'sub-spaceloc1006',
                             # 'sub-spaceloc1007', 'sub-spaceloc1008', 'sub-spaceloc1009',
                             # 'sub-spaceloc1010', 'sub-spaceloc1011', 'sub-spaceloc1012', 'sub-spaceloc2013']))
#]

rois = ['pIPS', 'LO']
hemispheres = ['left', 'right']
localizer = 'Tool'
run_combos = ['12', '21']

def combine_and_transform(subject):
    logging.info(f"Processing subject: {subject}")
    sub_dir = f'{study_dir}/{subject}/ses-01/'
    gca_dir = f'{sub_dir}/derivatives/gca'
    anat_brain = f"{raw_dir}/{subject}/ses-01/anat/{subject}_ses-01_T1w_brain.nii.gz"

    if not os.path.isfile(anat_brain):
        logging.error(f"Anatomical image not found for {subject}. Exiting...")
        return

    os.makedirs(gca_dir, exist_ok=True)

    anat2mni_mat = f"/user_data/csimmon2/temp_derivatives/{subject}/ses-01/derivatives/fc/anat2mni.mat"
    
    if not os.path.isfile(anat2mni_mat):
        logging.error(f"Transformation matrix not found for {subject}. Exiting...")
        return

    for roi in rois:
        for hemi in hemispheres:
            logging.info(f"Processing {roi} {hemi} for subject {subject}")

            all_run_results = []
            for run_combo in run_combos:
                img_path = f'{gca_dir}/searchlight_result_nontool_runs{run_combo}_{roi}_{hemi}_1217.nii.gz'
                #img_path = f'{gca_dir}/searchlight_result_tool_runs{run_combo}_{roi}_{hemi}_1217.nii.gz'
                if not os.path.exists(img_path):
                    logging.warning(f"File not found: {img_path}")
                    continue
                img = nib.load(img_path)
                all_run_results.append(img)

            if not all_run_results:
                logging.error(f"No valid images found for {roi} {hemi}")
                continue

            mean_img = image.mean_img(all_run_results)

            native_output_path = f'{gca_dir}/combined_nontool_{roi}_{hemi}_native_1217.nii.gz'
            #native_output_path = f'{gca_dir}/combined_tool_{roi}_{hemi}_native_1217.nii.gz'
            nib.save(mean_img, native_output_path)
            logging.info(f"Saved combined native image: {native_output_path}")

            mni_output_path = f'{gca_dir}/combined_nontool_{roi}_{hemi}_mni_1217.nii.gz'
            #mni_output_path = f'{gca_dir}/combined_tool_{roi}_{hemi}_mni_1217.nii.gz'
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

# to run one at a time
#if __name__ == "__main__":
    #if len(sys.argv) != 2:
        #print("Usage: python combine_and_transform_searchlight.py <subject>")
        #sys.exit(1)

   # subject = sys.argv[1]
   # combine_and_transform(subject)

# to run all    
if __name__ == "__main__":
    # Replace the command line argument check with direct processing of all subjects
    subjects = sub_info['sub'].unique()
    
    for subject in subjects:
        logging.info(f"Starting processing for subject: {subject}")
        try:
            combine_and_transform(subject)
        except Exception as e:
            logging.error(f"Error processing {subject}: {e}")
            continue