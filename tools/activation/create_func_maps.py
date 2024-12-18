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
import itertools
from nilearn import image, plotting, datasets
from nilearn.datasets import load_mni152_brain_mask, load_mni152_template
import nibabel as nib
import pdb
import os
import ptoc_params as params
import warnings
warnings.filterwarnings("ignore")

raw_dir = params.raw_dir
results_dir = params.results_dir
fig_dir = params.fig_dir
sub_info = params.sub_info_tool[params.sub_info_tool['exp'] == 'spaceloc']
task_info = params.task_info
thresh = params.thresh
suf = params.suf
mni = load_mni152_brain_mask()
roi_dir = '/user_data/csimmon2/git_repos/ptoc/roiParcels'

def create_sub_map():
    print('Creating individual subject maps...')
    
    for task, cond, cope in zip(task_info['task'], task_info['cond'], task_info['cope']):
        if task != 'toolloc' or cond != 'tool':
            continue
            
        for roi_type in ['ventral_visual_cortex', 'dorsal_visual_cortex']:
            # Load and binarize ROI
            roi = image.load_img(f'{roi_dir}/{roi_type}.nii.gz')
            roi = image.math_img('img > 0', img=roi)
            
            for sub in sub_info['sub']:
                sub_dir = f'{raw_dir}/{sub}/ses-01'
                zstat_path = f'{sub_dir}/derivatives/fsl/{task}/HighLevel{suf}.gfeat/cope{cope}.feat/stats/zstat1_reg.nii.gz'
                
                if os.path.exists(zstat_path):
                    print(f'Processing {sub} {cond}, {roi_type}')
                    os.makedirs(f'{sub_dir}/derivatives/neural_map', exist_ok=True)
                    
                    # Load and threshold zstat
                    zstat = image.load_img(zstat_path)
                    zstat = image.threshold_img(zstat, threshold=thresh, two_sided=False)
                    
                    # Get whole brain data first
                    whole_brain = zstat.get_fdata()
                    
                    # Mask zstat with ROI
                    zstat_masked = image.math_img('img1 * img2', img1=zstat, img2=roi)
                    func_np = zstat_masked.get_fdata()
                    
                    # Create binary versions
                    binary_3dfunc = np.copy(func_np)
                    binary_3dfunc[func_np>0] = 1
                    whole_brain[whole_brain>0] = 1
                    
                    # Average across voxels in z dimension
                    func_np = np.transpose(np.max(func_np, axis=2))
                    
                    # Create binary version of 2D map
                    binary_func = np.copy(func_np)
                    binary_func[binary_func>0] = 1
                    
                    # Save all versions
                    np.save(f'{sub_dir}/derivatives/neural_map/{cond}_func.npy', func_np)
                    np.save(f'{sub_dir}/derivatives/neural_map/{cond}_binary.npy', binary_func)
                    np.save(f'{sub_dir}/derivatives/neural_map/{cond}_binary_3d.npy', binary_3dfunc)
                    np.save(f'{sub_dir}/derivatives/neural_map/{cond}_whole_brain.npy', whole_brain)
                else:
                    print(f'{cond} zstat does not exist for subject {sub}')

def create_group_map():
    print('Creating group maps...')
    
    for task, cond, cope in zip(task_info['task'], task_info['cond'], task_info['cope']):
        if task != 'toolloc' or cond != 'tool':
            continue
            
        print(f'Processing {cond} {task}')
        func_list = []
        binary_list = []
        
        for sub in sub_info['sub']:
            sub_dir = f'{raw_dir}/{sub}/ses-01'
            neural_map_path = f'{sub_dir}/derivatives/neural_map/{cond}_func.npy'
            
            if os.path.exists(neural_map_path):
                # Load and normalize neural map
                neural_map = np.load(neural_map_path)
                neural_map = neural_map/np.max(neural_map)
                func_list.append(neural_map)
                
                # Load binary map
                binary_map = np.load(f'{sub_dir}/derivatives/neural_map/{cond}_binary.npy')
                binary_list.append(binary_map)
        
        if func_list and binary_list:
            # Average func maps across subjects
            func_group = np.nanmean(func_list, axis=0)
            
            # Sum binary maps
            binary_group = np.nansum(binary_list, axis=0)
            
            # Save results
            os.makedirs(f'{results_dir}/neural_map', exist_ok=True)
            np.save(f'{results_dir}/neural_map/{cond}_func.npy', func_group)
            np.save(f'{results_dir}/neural_map/{cond}_binary.npy', binary_group)

def create_3d_group_map():
    print('Creating 3d group maps...')
    
    for task, cond, cope in zip(task_info['task'], task_info['cond'], task_info['cope']):
        if task != 'toolloc' or cond != 'tool':
            continue
            
        print(f'Processing {cond} {task}')
        n = 0
        binary_list = []     
        
        for sub in sub_info['sub']:
            sub_dir = f'{raw_dir}/{sub}/ses-01'
            # Key difference: using binary_3d.npy files
            binary_map_path = f'{sub_dir}/derivatives/neural_map/{cond}_binary_3d.npy'

            if os.path.exists(binary_map_path):
                print(f'path exists for {sub}')
                if n == 0:
                    zstat_reg = image.load_img(f'{sub_dir}/derivatives/fsl/{task}/HighLevel{suf}.gfeat/cope{cope}.feat/stats/zstat1_reg.nii.gz')
                    affine = zstat_reg.affine
                    header = zstat_reg.header
                    n+=1

                binary_map = np.load(binary_map_path)
                binary_list.append(binary_map)

        if binary_list:
            binary_group = np.nansum(binary_list, axis=0)
            binary_group = nib.Nifti1Image(binary_group, affine, header)
            nib.save(binary_group, f'{results_dir}/neural_map/{cond}_group.nii.gz')

def create_raw_group_map():
    print('Creating raw group activation maps...')
    
    for task, cond, cope in zip(task_info['task'], task_info['cond'], task_info['cope']):
        if task != 'toolloc' or cond != 'tool':
            continue
            
        print(f'Processing {cond} {task}')
        n = 0
        activation_list = []
        
        for sub in sub_info['sub']:
            sub_dir = f'{raw_dir}/{sub}/ses-01'
            zstat_path = f'{sub_dir}/derivatives/fsl/{task}/HighLevel{suf}.gfeat/cope{cope}.feat/stats/zstat1_reg.nii.gz'
            
            if os.path.exists(zstat_path):
                print(f'Loading raw activation for {sub}')
                if n == 0:
                    # Store affine and header from first subject for later
                    zstat_img = image.load_img(zstat_path)
                    affine = zstat_img.affine
                    header = zstat_img.header
                    n += 1
                
                # Load raw zstat data without thresholding
                zstat_data = image.load_img(zstat_path).get_fdata()
                
                # Normalize within subject to account for scaling differences
                zstat_data = zstat_data / np.max(np.abs(zstat_data))
                
                activation_list.append(zstat_data)
            else:
                print(f'No zstat found for subject {sub}')
        
        if activation_list:
            # Average activation across subjects
            group_activation = np.mean(activation_list, axis=0)
            
            # Create and save nifti image
            group_img = nib.Nifti1Image(group_activation, affine, header)
            output_path = f'{results_dir}/neural_map/{cond}_group_raw_activation.nii.gz'
            nib.save(group_img, output_path)
            
            # Create visualization using nilearn
            plot_path = f'{fig_dir}/{cond}_group_activation.png'
            display = plotting.plot_stat_map(
                group_img,
                display_mode='ortho',
                cut_coords=(0, 0, 0),
                threshold=0.3,  # Adjust threshold as needed
                colorbar=True,
                title=f'Group Level {cond.capitalize()} Activation'
            )
            display.savefig(plot_path)
            display.close()
            
            print(f'Saved group activation map to {output_path}')
            print(f'Saved visualization to {plot_path}')
            
def main():
    #create_sub_map()
    #create_group_map()
    #create_3d_group_map()
    create_raw_group_map()

if __name__ == '__main__':
    main()