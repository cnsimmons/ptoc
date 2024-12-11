'''
Create 2D heatmap of activation for each subject and group average
'''
#use this create_func_map.py script to create individual and group-level functional maps for toolloc task
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
#hide warning
import warnings
warnings.filterwarnings("ignore")

raw_dir = params.raw_dir
results_dir = params.results_dir
fig_dir = params.fig_dir

sub_info = params.sub_info_tool[params.sub_info_tool['exp'] == 'spaceloc']
task_info = params.task_info  
thresh = params.thresh
suf = params.suf
rois = params.rois
hemis = params.hemis

mni = load_mni152_brain_mask()
roi_dir = '/user_data/csimmon2/git_repos/ptoc/roiParcels'

def create_sub_map():
    print('Creating individual subject maps...')
    for task, cond, cope in zip(task_info['task'], task_info['cond'], task_info['cope']):
        if task != 'toolloc' or cond != 'tool':
            continue
        
        roi_types = ['ventral_visual_cortex', 'dorsal_visual_cortex']
        
        for roi_type in roi_types:
            roi_path = f'{roi_dir}/{roi_type}.nii.gz'
            roi = image.load_img(roi_path)
            roi = image.math_img('img1 > 0', img1=roi)
            print(roi_path)
            print(roi.shape, roi.affine)
            
            for sub, code, hemi in zip(sub_info['sub'], sub_info['code'], sub_info['intact_hemi']):
                sub_dir = f'{raw_dir}/{sub}/ses-01'
                zstat_path = f'{sub_dir}/derivatives/fsl/{task}/HighLevel{suf}.gfeat/cope{cope}.feat/stats/zstat1_reg.nii.gz'
                
                if os.path.exists(zstat_path):
                    print(f'Processing {sub} {cond}, {roi_type}')
                    zstat = image.load_img(zstat_path)
                    zstat = image.threshold_img(zstat, threshold=thresh, two_sided=False)
                    whole_brain = zstat.get_fdata()
                    zstat_masked = image.math_img('img1 * img2', img1=zstat, img2=roi)
                    func_np = zstat_masked.get_fdata()
                    
                    binary_3dfunc = np.copy(func_np)
                    binary_3dfunc[func_np>0] = 1
                    whole_brain[whole_brain>0] = 1
                    
                    func_np = np.transpose(np.max(func_np, axis=2))
                    binary_func = np.copy(func_np)
                    binary_func[binary_func>0] = 1
                    
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
        
        roi_types = ['ventral_visual_cortex', 'dorsal_visual_cortex']
        
        for roi_type in roi_types:
            print(f'Processing {cond} {task} {roi_type}')
            func_list = []
            binary_list = []
            
            for sub in sub_info['sub']:
                sub_dir = f'{raw_dir}/{sub}/ses-01'
                neural_map_path = f'{sub_dir}/derivatives/neural_map/{cond}_func.npy'
                
                if os.path.exists(neural_map_path):
                    neural_map = np.load(neural_map_path)
                    neural_map = neural_map / np.max(neural_map)
                    func_list.append(neural_map)
                    
                    binary_map_path = f'{sub_dir}/derivatives/neural_map/{cond}_binary.npy'
                    if os.path.exists(binary_map_path):
                        binary_map = np.load(binary_map_path)
                        binary_list.append(binary_map)
            
            if func_list and binary_list:
                func_group = np.nanmean(func_list, axis=0)
                binary_group = np.nansum(binary_list, axis=0)
                
                os.makedirs(f'{results_dir}/neural_map', exist_ok=True)
                np.save(f'{results_dir}/neural_map/{cond}_func.npy', func_group)
                np.save(f'{results_dir}/neural_map/{cond}_binary.npy', binary_group)
            else:
                print(f'No valid neural maps found for condition {cond} and ROI {roi_type}.')

create_sub_map()
create_group_map()

