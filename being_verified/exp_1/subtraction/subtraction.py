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

def load_and_subtract_fc_maps(pIPS_path, LO_path):
    """
    Load pIPS and LO functional connectivity maps and subtract them.
    """
    pIPS_map = image.load_img(pIPS_path)
    LO_map = image.load_img(LO_path)
    
    difference_map = image.math_img("img1 - img2", img1=LO_map, img2=pIPS_map)
    return difference_map

def safe_divide(a, b):
    """
    Perform division safely, handling divide by zero and invalid value warnings.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        c[~np.isfinite(c)] = 0  # replace inf/NaN with 0
    return c

def subject_level_analysis(subject_id, pIPS_path, LO_path, output_dir, hemi):
    """
    Perform subject-level subtraction and save the result.
    """
    pIPS_file = pIPS_path.format(subject_id=subject_id, hemi=hemi)
    LO_file = LO_path.format(subject_id=subject_id, hemi=hemi)
    
    if not os.path.exists(pIPS_file) or not os.path.exists(LO_file):
        print(f"Warning: Files not found for subject {subject_id}, hemisphere {hemi}. Skipping.")
        return None

    difference_map = load_and_subtract_fc_maps(pIPS_file, LO_file)
    
    # Save the subtraction result in the subject's directory
    subtraction_file = os.path.join(output_dir, f'{subject_id}_LO_minus_pIPS_{hemi}_fc.nii.gz')
    difference_map.to_filename(subtraction_file)
    
    print(f"Subtraction complete for subject {subject_id}, hemisphere {hemi}. Result saved in {subtraction_file}")
    return subtraction_file

def group_level_analysis(subtraction_files, output_dir, hemi):
    """
    Perform group-level t-test and save the result.
    """
    if not subtraction_files:
        print(f"Error: No valid subtraction files for {hemi} hemisphere. Skipping t-test.")
        return

    # Load all subtraction maps
    all_maps = [image.load_img(f) for f in subtraction_files]
    
    # Stack all maps into a 4D image
    stacked_maps = image.concat_imgs(all_maps)

    # Perform voxel-wise one-sample t-test
    data = stacked_maps.get_fdata()
    mean = np.mean(data, axis=-1)
    std = np.std(data, axis=-1)
    n = data.shape[-1]
    t_values = safe_divide(mean, (std / np.sqrt(n)))

    # Create and save the t-map
    t_map = image.new_img_like(stacked_maps, t_values)
    t_map_file = os.path.join(output_dir, f'group_ttest_LO_minus_pIPS_{hemi}.nii.gz')
    t_map.to_filename(t_map_file)

    print(f"Group-level analysis complete for {hemi} hemisphere. T-map saved in {t_map_file}")

if __name__ == "__main__":
    # Define parameters
    
    # Load subject info
    sub_info = pd.read_csv(f'{curr_dir}/sub_info.csv')
    # Uncomment the next line to use all control subjects
    subs = sub_info[sub_info['group'] == 'control']['sub'].tolist()
    #subs = ['sub-025']  # For testing with a single subject
    
    study = 'ptoc'
    study_dir = f"/lab_data/behrmannlab/vlad/{study}"
    
    # Define paths for input files
    pIPS_path = f"{study_dir}/{{subject_id}}/ses-01/derivatives/fc/{{subject_id}}_pIPS_{{hemi}}_loc_fc.nii.gz"
    LO_path = f"{study_dir}/{{subject_id}}/ses-01/derivatives/fc/{{subject_id}}_LO_{{hemi}}_loc_fc.nii.gz"
    
    # Define output directories
    subject_out_dir = f"{study_dir}/{{subject_id}}/ses-01/derivatives/fc"
    group_out_dir = f'{curr_dir}/analyses/fc_subtraction'
    os.makedirs(group_out_dir, exist_ok=True)
    
    # Perform subject-level analysis and collect subtraction files
    subtraction_files = {'left': [], 'right': []}
    for subject_id in tqdm(subs, desc="Processing subjects"):
        for hemi in ['left', 'right']:
            subtraction_file = subject_level_analysis(
                subject_id, 
                pIPS_path, 
                LO_path, 
                subject_out_dir.format(subject_id=subject_id), 
                hemi
            )
            if subtraction_file:
                subtraction_files[hemi].append(subtraction_file)
    
    # Perform group-level analysis
    for hemi in ['left', 'right']:
        group_level_analysis(subtraction_files[hemi], group_out_dir, hemi)

print("Analysis completed for both hemispheres.")