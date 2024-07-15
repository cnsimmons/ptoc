#parameters for ptoc project

curr_dir = '/user_data/csimmon2/git_repos/ptoc'

import os
import shutil
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,curr_dir)
import subprocess
from glob import glob as glob
import pdb

runs = [1,2,3]
raw_dir = '/lab_data/behrmannlab/vlad/hemispace'
data_dir = '/lab_data/behrmannlab/vlad/ptoc'
results_dir = '/user_data/csimmon2/git_repos/ptoc/results'
fig_dir = '/user_data/csimmon2/git_repos/ptoc/results/figures'
sub_info = pd.read_csv(f'{curr_dir}/sub_info.csv')

hemis = ['left','right']

#pre 2024#rois = ['ventral_visual_cortex', 'dorsal_visual_cortex', 'LO', 'PFS', 'pIPS','aIPS']
rois = ['ventral_visual_cortex', 'dorsal_visual_cortex', 'LO', 'PFS', 'pIPS','aIPS', 'V1', 'hemi'] #adding V1 for early vision scramble and full hemispace || I need to add 'hemi' but at the moment I can't find the roi file so I'm leaving as is with just the added V1, i added hemi after running the register_mirror.py script


thresh = 2.58

#task = pd.read_csv(f'{curr_dir}/task_info.csv') #from vlads params originally, created task_info file 7/15/24. Task_info is a simple CSV with task condition and cope
task = 'loc'
task_info = 'loc' #pre 2024# 
cond = 'scramble' #'object' #'scramble'
cope = '5' #3 #5 #short cut until I can recode scripts for task_info csv instead of the discrete
suf = ''

tr = 2
vols = 184

