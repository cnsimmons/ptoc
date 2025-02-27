#loops over analysis scripts and runs them

curr_dir = '/user_data/csimmon2/git_repos/ptoc'

import os
import shutil
import sys
import numpy as np
import pandas as pd

sys.path.insert(0,curr_dir)

import subprocess
from glob import glob as glob
import ptoc_params as params

curr_script = f'{curr_dir}/fmri/pre_proc/preprocess.py'

#curr_script = f'{curr_dir}/fmri/pre_proc/register_1stlevel.py' ##This script register the time series data from each run (the 1stlevel) to that individual's anatomical

sub_info = params.sub_info #run all subjects

#sub_info = sub_info.head(4) #DELETE THIS WHEN YOU RUN THE WHOLE THING
target_subject = sub_info['sub'].iloc[14] #this is the subject that I want to run the script on. I can change this to any subject in the sub_info dataframe.

print(target_subject) 

#for sub in sub_info['sub']:
    #try:
     #   bash_cmd = f'python {curr_script} {sub}'
      #  subprocess.run(bash_cmd.split(),check = True)
    #except:
    #    print(f'failed on {sub}')
    
      
try:
    bash_cmd = f'python {curr_script} {target_subject}'
    subprocess.run(bash_cmd.split(),check = True)
except:
    print(f'failed on {target_subject}')

