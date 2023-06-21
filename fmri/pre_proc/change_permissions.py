#create a script the globs all folders in a directory and then changes their group to behrmanlab using subprocess
#run in parallel
curr_dir = '/user_data/vayzenbe/GitHub_Repos/ptoc'
import sys
sys.path.append(curr_dir)
import subprocess
import os
from glob import glob as glob
import ptoc_params as params
import pdb



#set the directory to the current working directory
target_dir = params.raw_dir

#set the group to behrmanlab
group = 'behrmannlab'

#set the permissions to 770
permissions = '770'

#glob all the folders in the target directory
folders = glob(f'{target_dir}/*')


#loop through the folders and change the group and permissions in parallel
import multiprocessing as mp
def change_group_permissions(folder):
    print(folder.split('/')[-1])

    subprocess.run(['chgrp', '-R', group, folder])
    subprocess.run(['chmod', '-R', permissions, folder])
    return



pool = mp.Pool(mp.cpu_count())
pool.map(change_group_permissions, folders)

