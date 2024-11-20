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

suf = ''

#runs = [1,2,3]
runs = [1,2] #change to 2 for sub_057
raw_dir = '/lab_data/behrmannlab/vlad/hemispace'
data_dir = '/lab_data/behrmannlab/vlad/ptoc'
results_dir = '/user_data/csimmon2/git_repos/ptoc/results/{task}'
fig_dir = '/user_data/csimmon2/git_repos/ptoc/results/figures'
sub_info = pd.read_csv(f'{curr_dir}/sub_info.csv')
sub_info_tool = pd.read_csv(f'{curr_dir}/sub_info_tool.csv')
sub_info_OTC = pd.read_csv(f'{curr_dir}/sub_info_OTC.csv')
task_info = pd.read_csv(f'/user_data/csimmon2/git_repos/ptoc/task_info{suf}.csv')

hemis = ['left','right']
rois = ['ventral_visual_cortex', 'dorsal_visual_cortex', 'LO', 'PFS', 'pIPS','aIPS', 'V1'] #adding V1 for early vision scramble and full hemispace || I need to add 'hemi' but at the moment I can't find the roi file so I'm leaving as is with just the added V1

##temporary
#cope = '4' #3 #5 #short cut until I can recode scripts for task_info csv instead of the discrete || i thought 3 is for object and 5 is for scramble but 4 is for scramble and 5 is for word? Needs to verify. 
#task_info = 'loc' #pre 2024# 
#cond = '' #'object' #'scramble'
task = 'toolloc'

thresh = 2.58

tr = 2
vols = 184

#BWOC Params

'''
cov_dir = f'{params.loc_data}/sub-{sub}/ses-01/covs'
loc_data = f'/lab_data/behrmannlab/vlad/hemispace
'''