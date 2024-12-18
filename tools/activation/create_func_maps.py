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

import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import itertools
from nilearn import image, plotting, datasets
from nilearn.datasets import load_mni152_brain_mask, load_mni152_template
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
task_info = params.task_info
thresh = params.thresh
suf = params.suf
mni = load_mni152_brain_mask()
roi_dir = '/user_data/csimmon2/git_repos/ptoc/roiParcels'

def create_sub_map():
    print('Creating individual subject maps...')
    
    for task, cond, cope in zip(task_info['task'], task_info['cond'], task_info['cope']):
        # Only process tool localization
        if task != 'toolloc' or cond != 'tool':
            continue
            
        print(f'Processing {task} {cond}')
        
        for sub in sub_info['sub']:
            sub_dir = f'{raw_dir}/{sub}/ses-01'
            zstat_path = f'{sub_dir}/derivatives/fsl/{task}/HighLevel{suf}.gfeat/cope{cope}.feat/stats/zstat1_reg.nii.gz'
            
            if os.path.exists(zstat_path):
                print(f'Processing subject {sub}')
                os.makedirs(f'{sub_dir}/derivatives/neural_map', exist_ok=True)
                
                # Load and threshold zstat
                zstat = image.load_img(zstat_path)
                zstat = image.threshold_img(zstat, threshold=thresh, two_sided=False)
                
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
    print('Creating group maps...')
    
    for task, cond, cope in zip(task_info['task'], task_info['cond'], task_info['cope']):
        # Only process tool localization
        if task != 'toolloc' or cond != 'tool':
            continue
            
        print(f'Processing {cond} {task}')
        n = 0
        binary_list = []
        
        for sub in sub_info['sub']:
            sub_dir = f'{raw_dir}/{sub}/ses-01'
            binary_map_path = f'{sub_dir}/derivatives/neural_map/{cond}_binary_3d.npy'
            
            if os.path.exists(binary_map_path):
                print(f'Processing subject {sub}')
                if n == 0:
                    # Get affine and header from first subject
                    zstat_reg = image.load_img(f'{sub_dir}/derivatives/fsl/{task}/HighLevel{suf}.gfeat/cope{cope}.feat/stats/zstat1_reg.nii.gz')
                    affine = zstat_reg.affine
                    header = zstat_reg.header
                    n += 1
                
                # Load binary map
                binary_map = np.load(binary_map_path)
                binary_list.append(binary_map)
        
        if binary_list:
            print(f'Creating group map from {len(binary_list)} subjects')
            # Sum binary maps
            binary_group = np.sum(binary_list, axis=0)
            
            # Save results
            os.makedirs(f'{results_dir}/neural_map', exist_ok=True)
            np.save(f'{results_dir}/neural_map/{cond}_group_map.npy', binary_group)
            
            # Save as nifti
            binary_group_nii = nib.Nifti1Image(binary_group, affine, header)
            nib.save(binary_group_nii, f'{results_dir}/neural_map/{cond}_group.nii.gz')
            
            print(f'Maximum overlap: {np.max(binary_group)} subjects')

def main():
    create_sub_map()
    create_group_map()

if __name__ == '__main__':
    main()