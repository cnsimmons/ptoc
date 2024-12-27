'''
Create 2D heatmap of activation for each subject and group average
'''
curr_dir = f'/user_data/csimmon2/git_repos/ptoc'

import sys
sys.path.insert(0,curr_dir)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from nilearn import image, plotting, datasets
from nilearn.datasets import load_mni152_brain_mask
import nibabel as nib
import os
import ptoc_params as params
import warnings
warnings.filterwarnings("ignore")

# Paths and parameters
raw_dir = params.raw_dir
results_dir = params.results_dir
fig_dir = params.fig_dir
sub_info = params.sub_info_tool[params.sub_info_tool['exp'] == 'spaceloc']
thresh = params.thresh
suf = params.suf
roi_dir = params.roi_dir


def create_sub_map():
    print('Creating individual subject maps...')
    
    task = 'toolloc'
    cond = 'tool'
    cope = 1

    # Determine ROI type
    if cond == 'word' or task == 'face':
        roi_type = 'ventral_visual_cortex'
    elif cond == 'tool' or cond == 'space':
        roi_type = 'dorsal_visual_cortex'
    
    # Load and binarize ROI
    roi = image.load_img(f'{roi_dir}/{roi_type}.nii.gz')
    roi = image.math_img('img > 0', img=roi)
            
    for sub in sub_info['sub']:
        sub_dir = f'{raw_dir}/{sub}/ses-01'
        zstat_path = f'{sub_dir}/derivatives/fsl/{task}/HighLevel{suf}.gfeat/cope{cope}.feat/stats/zstat1_reg.nii.gz'
        
        if os.path.exists(zstat_path):
            print(f'Processing subject {sub}')
            os.makedirs(f'{sub_dir}/derivatives/neural_map', exist_ok=True)
            
            # Load and threshold zstat
            zstat = image.load_img(zstat_path)
            zstat = image.threshold_img(zstat, threshold=thresh, two_sided=False)
            
            # Apply ROI mask
            zstat_masked = image.math_img('img1 * img2', img1=zstat, img2=roi)
            
            # Get masked functional data
            func_np = zstat_masked.get_fdata()
            
            # Get whole brain data before masking
            whole_brain = zstat.get_fdata()
            whole_brain[whole_brain > 0] = 1
            
            # Create binary 3D version
            binary_3dfunc = np.copy(func_np)
            binary_3dfunc[func_np > 0] = 1
            
            # Create 2D version (max across z-axis)
            func_2d = np.transpose(np.max(func_np, axis=2))
            
            # Create binary 2D version
            binary_func = np.copy(func_2d)
            binary_func[binary_func > 0] = 1
            
            # Save all versions
            np.save(f'{sub_dir}/derivatives/neural_map/{cond}_func.npy', func_2d)
            np.save(f'{sub_dir}/derivatives/neural_map/{cond}_binary.npy', binary_func)
            np.save(f'{sub_dir}/derivatives/neural_map/{cond}_binary_3d.npy', binary_3dfunc)
            np.save(f'{sub_dir}/derivatives/neural_map/{cond}_whole_brain.npy', whole_brain)
        else:
            print(f'No zstat found for subject {sub}')

def create_group_map():
    print('Creating 2D group maps...')
    
    task = 'toolloc'
    cond = 'tool'
    cope = 1
    
    n = 0
    func_list = []
    binary_list = []
    
    for sub in sub_info['sub']:
        sub_dir = f'{raw_dir}/{sub}/ses-01'
        neural_map_path = f'{sub_dir}/derivatives/neural_map/{cond}_func.npy'
        
        if os.path.exists(neural_map_path):
            print(f'Processing subject {sub}')
            
            # Load neural map
            neural_map = np.load(neural_map_path)
            
            # Normalize individual maps
            if np.max(neural_map) > 0:
                neural_map = neural_map / np.max(neural_map)
            func_list.append(neural_map)
            
            # Load binary map
            binary_map = np.load(f'{sub_dir}/derivatives/neural_map/{cond}_binary.npy')
            binary_list.append(binary_map)
    
    if binary_list:
        # Diagnostic prints
        for i, binary_map in enumerate(binary_list):
            active_voxels = np.sum(binary_map > 0)
            print(f"Subject {i+1}: {active_voxels} active voxels")
        
        # Create group maps
        func_group = np.nanmean(func_list, axis=0)
        binary_group = np.sum(binary_list, axis=0)
        print(f"\nGroup map stats:")
        print(f"Max overlap: {np.max(binary_group)} subjects")
        print(f"Number of voxels with any activation: {np.sum(binary_group > 0)}")
        
        # Save group maps
        os.makedirs(f'{results_dir}/neural_map', exist_ok=True)
        np.save(f'{results_dir}/neural_map/{cond}_func.npy', func_group)
        np.save(f'{results_dir}/neural_map/{cond}_binary.npy', binary_group)

def create_3d_group_map():
    print('Creating 3D group maps...')
    
    task = 'toolloc'
    cond = 'tool'
    cope = 1

    # Determine ROI type
    if cond == 'word' or task == 'face':
        roi_type = 'ventral_visual_cortex'
    elif cond == 'tool' or cond == 'space':
        roi_type = 'dorsal_visual_cortex'
    
    # Load and binarize ROI
    roi = image.load_img(f'{roi_dir}/{roi_type}.nii.gz')
    roi = image.math_img('img > 0', img=roi)
    
    n = 0
    binary_list = []
    
    for sub in sub_info['sub']:
        sub_dir = f'{raw_dir}/{sub}/ses-01'
        zstat_path = f'{sub_dir}/derivatives/fsl/{task}/HighLevel{suf}.gfeat/cope{cope}.feat/stats/zstat1_reg.nii.gz'
        
        if os.path.exists(zstat_path):
            print(f'Processing subject {sub}')
            
            if n == 0:
                # Get affine and header from first subject
                zstat_reg = image.load_img(zstat_path)
                affine = zstat_reg.affine
                header = zstat_reg.header
                n += 1
            
            # Load and threshold zstat
            zstat = image.load_img(zstat_path)
            zstat = image.threshold_img(zstat, threshold=thresh, two_sided=False)
            
            # Apply ROI mask
            zstat_masked = image.math_img('img1 * img2', img1=zstat, img2=roi)
            
            # Get masked data and binarize
            func_data = zstat_masked.get_fdata()
            binary_data = np.copy(func_data)
            binary_data[func_data > 0] = 1
            
            binary_list.append(binary_data)
    
    if binary_list:
        print(f'Creating 3D group map from {len(binary_list)} subjects')
        
        # Sum binary maps
        binary_group = np.nansum(binary_list, axis=0)
        
        # Save numpy array
        os.makedirs(f'{results_dir}/neural_map', exist_ok=True)
        np.save(f'{results_dir}/neural_map/{cond}_group_map.npy', binary_group)
        
        # Save as NIfTI
        print(f"Maximum overlap: {np.max(binary_group)} subjects")
        
        # Save as NIfTI
        binary_group = nib.Nifti1Image(binary_group, affine, header)
        nib.save(binary_group, f'{results_dir}/neural_map/{cond}_group.nii.gz')
        print(f"Saved 3D group maps for {cond}")

def main():
    create_sub_map()
    create_group_map()
    create_3d_group_map()

if __name__ == '__main__':
    main()