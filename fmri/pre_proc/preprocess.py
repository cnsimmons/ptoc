import os
import shutil
import sys
import subprocess
from glob import glob as glob
import pdb


exp = sys.argv[1]
sub = sys.argv[2]
print(exp, sub, flush = True)


#add exp to each item in sub_list
#

tasks = ['spaceloc', 'toolloc','loc']

ses = 1
runs = [1,2,3]

study_dir = f'/lab_data/behrmannlab/vlad/{exp}'


#instatiate sub dir
sub_dir = f'{study_dir}/sub-{sub}/ses-{ses:02d}'
#make derivatives folder
os.makedirs(f'{sub_dir}/derivatives', exist_ok=True)
#make fsl folder
os.makedirs(f'{sub_dir}/derivatives/fsl', exist_ok=True)
#make covs folder
os.makedirs(f'{sub_dir}/covs', exist_ok=True)


#extract anat files from anat folder
anat_file = glob(f'{sub_dir}/anat/*_T1w.nii.gz')[0]


#check if deskulled brain exists, else make it
if not os.path.exists(f'{sub_dir}/anat/sub-{sub}_ses-{ses:02d}_T1w_brain.nii.gz'):
    #deskull brain
    bash_cmd = f'bet {anat_file} {sub_dir}/anat/sub-{sub}_ses-{ses:02d}_T1w_brain.nii.gz -R -B'
    subprocess.run(bash_cmd.split(),check = True)

for task in tasks:
    #make task folder
    os.makedirs(f'{sub_dir}/derivatives/fsl/{task}', exist_ok=True)

    #loop through runs and calculate motion outliers
    for run in runs:
        #if bold file exists
        if os.path.exists(f'{sub_dir}/func/sub-{sub}_ses-{ses:02d}_task-{task}_run-0{run}_bold.nii.gz'):
            #create run directory
            os.makedirs(f'{sub_dir}/derivatives/fsl/{task}/run-0{run}', exist_ok=True)

            #calcualte motion outliers
            bash_cmd = f'fsl_motion_outliers -i {sub_dir}/func/sub-{sub}_ses-{ses:02d}_task-{task}_run-0{run}_bold.nii.gz -o {sub_dir}/derivatives/fsl/{task}/run-0{run}/sub-{sub}_ses-{ses:02d}_task-{task}_run-0{run}_bold_spikes.txt --dummy=0'
            subprocess.run(bash_cmd.split(),check = True)
            #print(bash_cmd)



