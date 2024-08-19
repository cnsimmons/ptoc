import sys
curr_dir = '/user_data/csimmon2/git_repos/ptoc'
sys.path.insert(0, curr_dir)

import numpy as np
from nilearn import image
from scipy import stats
import pandas as pd
import os
from tqdm import tqdm

def load_and_subtract_fc_maps(subject_id, pIPS_path, LO_path):
    """
    Load pIPS and LO functional connectivity maps and subtract them.
    """
    pIPS_map = image.load_img(pIPS_path)
    LO_map = image.load_img(LO_path)
    
    difference_map = image.math_img("img1 - img2", img1=LO_map, img2=pIPS_map)
    return difference_map

def run_analysis(subs, pIPS_path, LO_path, output_dir, hemi):
    """
    Run the analysis for all subjects and perform a t-test for a specific hemisphere.
    """
    all_difference_maps = []

    # Calculate difference maps for each subject
    for subject_id in tqdm(subs, desc=f"Processing subjects for {hemi} hemisphere"):
        pIPS_file = pIPS_path.format(subject_id=subject_id, hemi=hemi)
        LO_file = LO_path.format(subject_id=subject_id, hemi=hemi)
        difference_map = load_and_subtract_fc_maps(subject_id, pIPS_file, LO_file)
        all_difference_maps.append(difference_map)

    # Stack all difference maps into a 4D image
    difference_4d = image.concat_imgs(all_difference_maps)

    # Perform voxel-wise one-sample t-test
    t_map = image.math_img("np.mean(img, axis=-1) / (np.std(img, axis=-1) / np.sqrt(img.shape[-1]))",
                           img=difference_4d)

    # Save the results
    t_map.to_filename(os.path.join(output_dir, f'group_ttest_LO_minus_pIPS_{hemi}.nii.gz'))

    print(f"Analysis complete for {hemi} hemisphere. T-map saved in the output directory.")

if __name__ == "__main__":
    # Define parameters
    
    # Load subject info
    sub_info = pd.read_csv(f'{curr_dir}/sub_info.csv')
    # Uncomment the next line to use all control subjects
    # subs = sub_info[sub_info['group'] == 'control']['sub'].tolist()
    subs = ['sub-025']  # For testing with a single subject
    
    study = 'ptoc'
    study_dir = f"/lab_data/behrmannlab/vlad/{study}"
    
    # Define paths for input files
    fc_dir = f'{study_dir}/group/fc'
    pIPS_path = f"{fc_dir}/{{subject_id}}_pIPS_{{hemi}}_loc_fc.nii.gz"
    LO_path = f"{fc_dir}/{{subject_id}}_LO_{{hemi}}_loc_fc.nii.gz"
    
    # Create output directory
    out_dir = f'{fc_dir}/subtraction'
    os.makedirs(out_dir, exist_ok=True)
    
    # Run the analysis for both hemispheres
    for hemi in ['left', 'right']:
        run_analysis(subs, pIPS_path, LO_path, out_dir, hemi)

print("Analysis completed for both hemispheres.")