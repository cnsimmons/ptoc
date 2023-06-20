curr_dir = '/user_data/csimmon2/git_repos/ptoc'

import os
import shutil
import sys
sys.path.insert(0,curr_dir)
import subprocess
from glob import glob as glob
import pandas as pd

#command line options

sub = sys.argv[1] #WHICH SUBJECT

print(sub, flush = True)


#add exp to each item in sub_list

tasks = ['loc']

ses = 1 #this may need to change for which sesion you want to run? - Claire
runs = [1,2,3]

data_dir = f'/lab_data/behrmannlab/vlad/hemispace'
output_dir = f'/lab_data/behrmannlab/vlad/ptoc'

#instatiate sub dir
sub_data = f'{data_dir}/{sub}/ses-{ses:02d}'
sub_output = f'{output_dir}/{sub}/ses-{ses:02d}'

#make derivatives folder
os.makedirs(f'{sub_output}/derivatives', exist_ok=True)
#make fsl folder
os.makedirs(f'{sub_output}/derivatives/fsl', exist_ok=True)



#extract anat files from anat folder
anat_file = glob(f'{sub_data}/anat/*_T1w.nii.gz')[0]


#check if deskulled brain exists, else make it
if not os.path.exists(f'{sub_data}/anat/{sub}_ses-{ses:02d}_T1w_brain.nii.gz') and os.path.exists(anat_file):
    print('deskulling brain', flush = True)
    #deskull brain
    bash_cmd = f'bet {anat_file} {sub_data}/anat/{sub}_ses-{ses:02d}_T1w_brain.nii.gz -R -B'
    subprocess.run(bash_cmd.split(),check = True)

for task in tasks:
    print(task, flush = True)


    #make task folder
    os.makedirs(f'{sub_output}/derivatives/fsl/{task}', exist_ok=True)

    #loop through runs and calculate motion outliers
    for run in runs:
        #if bold file exists
        if os.path.exists(f'{sub_data}/func/{sub}_ses-{ses:02d}_task-{task}_run-0{run}_bold.nii.gz'):
            #create run directory
            os.makedirs(f'{sub_output}/derivatives/fsl/{task}/run-0{run}', exist_ok=True)

            #calcualte motion outliers
            bash_cmd = f'fsl_motion_outliers -i {sub_data}/func/{sub}_ses-{ses:02d}_task-{task}_run-0{run}_bold.nii.gz -o {sub_output}/derivatives/fsl/{task}/run-0{run}/{sub}_ses-{ses:02d}_task-{task}_run-0{run}_bold_spikes.txt --dummy=0'
            subprocess.run(bash_cmd.split(),check = True)
            #print(bash_cmd)



