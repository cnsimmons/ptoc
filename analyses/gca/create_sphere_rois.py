#creates spherical rois in registered native space

import os
import sys
import logging
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import image

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import your parameters
curr_dir = f'/user_data/csimmon2/git_repos/ptoc'
sys.path.insert(0, curr_dir)
import ptoc_params as params

# Set up directories and parameters
study = 'ptoc'
study_dir = f"/lab_data/behrmannlab/vlad/{study}"
localizer = 'Object'  # scramble or object. This is the localizer task.
results_dir = '/user_data/csimmon2/git_repos/ptoc/results'
raw_dir = params.raw_dir

# Load subject information
sub_info = pd.read_csv(f'{curr_dir}/sub_info.csv')
sub_info = sub_info[sub_info['group'] == 'control']
subs = sub_info['sub'].tolist()
#subs = ['sub-025']  # You can change this to sub_info['sub'].tolist() if needed

# ROI parameters
rois = ['pIPS', 'LO']
hemispheres = ['left', 'right']
radius = 6

def create_spherical_roi(coords, radius, affine, shape):
    """Create a spherical ROI mask."""
    coords = np.array(coords)
    mask = np.zeros(shape, dtype=bool)
    
    # Get voxel coordinates
    xx, yy, zz = np.meshgrid(np.arange(shape[0]),
                             np.arange(shape[1]),
                             np.arange(shape[2]),
                             indexing='ij')
    vox_coords = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    
    # Transform voxel coordinates to world coordinates
    world_coords = nib.affines.apply_affine(affine, vox_coords)
    
    # Calculate distances
    distances = np.sqrt(np.sum((world_coords - coords)**2, axis=1))
    
    # Create mask
    mask = distances <= radius
    return mask.reshape(shape)

def save_spherical_rois(ss, tsk='loc'):
    """
    Save spherical ROIs as NIfTI files based on coordinates.
    
    Parameters:
    - ss: str, subject ID
    - tsk: str, task name (default: 'loc')
    """
    logging.info(f"Processing subject: {ss}")
    
    # Define paths
    sub_dir = f'{study_dir}/{ss}/ses-01/'
    roi_dir = f'{sub_dir}derivatives/rois'
    out_dir = f'{sub_dir}derivatives/rois/spheres_nifti'
    os.makedirs(out_dir, exist_ok=True)
    
    # Read ROI coordinates
    roi_coords_file = f'{roi_dir}/spheres/sphere_coords_hemisphere.csv'
    roi_coords = pd.read_csv(roi_coords_file)
    
    # Get subject-specific brain mask to use as a template
    mask_path = f'{raw_dir}/{ss}/ses-01/anat/{ss}_ses-01_T1w_brain_mask.nii.gz'
    template = nib.load(mask_path)
    
    # Iterate through each ROI and hemisphere
    for rr in rois:
        for hemi in hemispheres:
            curr_coords = roi_coords[(roi_coords['task'] == tsk) & 
                                     (roi_coords['roi'] == rr) &
                                     (roi_coords['hemisphere'] == hemi)]
            
            if curr_coords.empty:
                logging.warning(f"No coordinates found for {rr}, {hemi}")
                continue
            
            coords = curr_coords[['x', 'y', 'z']].values.tolist()[0]
            
            # Create spherical ROI
            roi_mask = create_spherical_roi(coords, radius, template.affine, template.shape)
            roi_img = nib.Nifti1Image(roi_mask.astype(np.int16), template.affine, template.header)
            
            # Save ROI as NIfTI file
            output_file = os.path.join(out_dir, f"{ss}_{rr}_{hemi}_{tsk}_sphere_r{radius}mm.nii.gz")
            nib.save(roi_img, output_file)
            logging.info(f"Saved ROI: {output_file}")

# Main execution
if __name__ == "__main__":
    for ss in subs:
        save_spherical_rois(ss)

logging.info("ROI saving process completed.")