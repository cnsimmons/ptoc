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

def create_sub_map():
    print('Creating individual subject maps...')
    
    task = 'toolloc'
    cond = 'tool'
    cope = 1
            
    for sub in sub_info['sub']:
        sub_dir = f'{raw_dir}/{sub}/ses-01'
        zstat_path = f'{sub_dir}/derivatives/fsl/{task}/HighLevel{suf}.gfeat/cope{cope}.feat/stats/zstat1_reg.nii.gz'
        
        if os.path.exists(zstat_path):
            print(f'Processing subject {sub}')
            os.makedirs(f'{sub_dir}/derivatives/neural_map', exist_ok=True)
            
            # Load and threshold zstat
            zstat = image.load_img(zstat_path)
            
            # After loading zstat
            print(f"Before threshold: max={np.max(zstat.get_fdata())}")
            zstat = image.threshold_img(zstat, threshold=thresh, two_sided=False)
            print(f"After threshold: max={np.max(zstat.get_fdata())}")
            
            # Get whole brain data
            whole_brain = zstat.get_fdata()
            whole_brain[whole_brain > 0] = 1
            
            # Create binary version of 3D data
            func_np = zstat.get_fdata()
            binary_3dfunc = np.copy(func_np)
            binary_3dfunc[func_np > 0] = 1
            
            # Save binary maps
            np.save(f'{sub_dir}/derivatives/neural_map/{cond}_binary_3d.npy', binary_3dfunc)
            np.save(f'{sub_dir}/derivatives/neural_map/{cond}_whole_brain.npy', whole_brain)
            
        else:
            print(f'No zstat found for subject {sub}')

def create_group_map():
    """
    Creates 2D group-level maps from individual subject data.
    Includes proportion-based normalization for functional data.
    """
    print('Creating 2D group maps...')
    
    task = 'toolloc'
    cond = 'tool'
    cope = 1
    
    n = 0
    func_list = []
    binary_list = []
    
    for sub in sub_info['sub']:
        sub_dir = f'{raw_dir}/{sub}/ses-01'
        zstat_path = f'{sub_dir}/derivatives/fsl/{task}/HighLevel{suf}.gfeat/cope{cope}.feat/stats/zstat1_reg.nii.gz'
        
        if os.path.exists(zstat_path):
            print(f'Processing subject {sub}')
            
            # Load zstat image
            zstat = image.load_img(zstat_path)
            
            # Store affine and header from first subject
            if n == 0:
                affine = zstat.affine
                header = zstat.header
                n += 1
            
            # Get functional data and threshold it
            zstat_thresh = image.threshold_img(zstat, threshold=thresh, two_sided=False)
            func_data = zstat_thresh.get_fdata()
            
            # Convert to 2D by taking maximum across z-dimension
            func_data = np.transpose(np.max(func_data, axis=2))
            
            # Rescale as proportion of max to normalize across subject activation
            if np.max(func_data) > 0:  # Avoid division by zero
                func_data = func_data / np.max(func_data)
            func_list.append(func_data)
            
            # Create binary version
            binary_data = np.copy(func_data)
            binary_data[binary_data > 0] = 1
            binary_list.append(binary_data)
        else:
            print(f'No zstat found for subject {sub}')
    
    if binary_list:
        n_subjects = len(binary_list)
        print(f'Creating group maps from {n_subjects} subjects')
        
        # Ensure output directory exists
        os.makedirs(f'{results_dir}/neural_map', exist_ok=True)
        
        # Process and save functional group average
        func_group = np.nanmean(func_list, axis=0)
        np.save(f'{results_dir}/neural_map/{cond}_func.npy', func_group)
        
        # Process and save binary group map
        binary_group = np.sum(binary_list, axis=0)
        np.save(f'{results_dir}/neural_map/{cond}_binary.npy', binary_group)
        
        print(f'Saved 2D functional and binary group maps for {cond}')

def create_3d_group_map():
    """
    Creates 3D group-level maps focusing on binary overlap maps.
    No proportion-based normalization, just binary summation.
    """
    print('Creating 3D group maps...')
    
    task = 'toolloc'
    cond = 'tool'
    cope = 1
    
    n = 0
    binary_list = []
    
    for sub in sub_info['sub']:
        sub_dir = f'{raw_dir}/{sub}/ses-01'
        binary_map_path = f'{sub_dir}/derivatives/neural_map/{cond}_binary_3d.npy'
        
        if os.path.exists(binary_map_path):
            print(f'Processing subject {sub}')
            
            if n == 0:
                # Get affine and header from first subject's zstat
                zstat_reg = image.load_img(f'{sub_dir}/derivatives/fsl/{task}/HighLevel{suf}.gfeat/cope{cope}.feat/stats/zstat1_reg.nii.gz')
                affine = zstat_reg.affine
                header = zstat_reg.header
                n += 1
            
            # Load binary map
            binary_map = np.load(binary_map_path)
            binary_list.append(binary_map)
        else:
            print(f'No binary map found for subject {sub}')
    
    if binary_list:
        n_subjects = len(binary_list)
        print(f'Creating 3D group map from {n_subjects} subjects')
        
        # Ensure output directory exists
        os.makedirs(f'{results_dir}/neural_map', exist_ok=True)
        
        # Sum binary maps
        binary_group = np.sum(binary_list, axis=0)
        
        # Save numpy array
        np.save(f'{results_dir}/neural_map/{cond}_group_map.npy', binary_group)
        
        # Save as NIfTI
        binary_group_nii = nib.Nifti1Image(binary_group, affine, header)
        nib.save(binary_group_nii, f'{results_dir}/neural_map/{cond}_group.nii.gz')
        
        print(f'Maximum overlap: {np.max(binary_group)} subjects')
        print(f'Saved 3D group maps for {cond}')

def main():
    create_sub_map()
    create_group_map()
    create_3d_group_map()

if __name__ == '__main__':
    main()