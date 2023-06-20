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
import pdb
import ptoc_params as params

curr_script = f'{curr_dir}/fmri/pre_proc/preprocess.py'

sub_info = params.sub_info
#pdb.set_trace()
sub_info = sub_info.head(4) #DELETE THIS WHNE YOU RUN THE WHOLE THING

for sub in sub_info['sub']:
    try:
        bash_cmd = f'python {curr_script} {sub}'
        subprocess.run(bash_cmd.split(),check = True)
    except:
        print(f'failed on {sub}')
            
