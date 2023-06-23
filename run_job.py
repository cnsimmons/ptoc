import subprocess
from glob import glob
import os
import time
import pdb
import pandas as pd

job_name = 'fsl_job'
mem = 24
run_time = "1-00:00:00"

pause_crit = 12
pause_time = 5

runs=list(range(1,3))
exp = 'hemispace'
sub_list = list(range(1001,1013)) + list(range(2013,2019))
sub_list = [f'spaceloc{sub}' for sub in sub_list]
tasks = ['spaceloc','toolloc', 'loc']

tasks = ['toolloc']



#sub_list = [25,38,57,59,64,67,68,71,83,84,85,87,88]
#sub_list = [f'0{sub}' if sub < 100 else f'{sub}' for sub in sub_list]
sub_list=["025", "038", "057", "059", "064", "067", "068", "071", "083", "084", "085", 
"087", "088", "093", "094", "095", "096", "097", "103", "104", "105", "106", "107", 
"hemispace1001", "hemispace1002", "hemispace1003","hemispace1004", "hemispace1006", "hemispace1007",
 "hemispace2001", "hemispace2002", "hemispace2003"]


sub_info = pd.read_csv('/user_data/vayzenbe/GitHub_Repos/hemispace/sub_info.csv')
sub_list = sub_info['sub'].tolist()
print(sub_list)
#sub_list=["sub-hemispace1002"]
#subj_list=["sub-057", "sub-068", "sub-095", "sub-104", "sub-105", "sub-spaceloc1005", "sub-spaceloc1006", "sub-spaceloc1011", "sub-spaceloc1012", "sub-spaceloc2017", "sub-spaceloc2018"]


study_dir= f'/lab_data/behrmannlab/vlad/{exp}'
ses = 1
suf = '_roi'

#the sbatch setup info
run_1stlevel = False
run_highlevel =True

preprocess = False



def setup_sbatch(job_name, script_name):
    sbatch_setup = f"""#!/bin/bash -l

# Job name
#SBATCH --job-name={job_name}

#SBATCH --mail-type=ALL
#SBATCH --mail-user=vayzenb@cmu.edu

# Submit job to cpu queue                
#SBATCH -p cpu

#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0
#SBATCH --exclude=mind-0-15,mind-0-14,mind-0-16
# Job memory request
#SBATCH --mem={mem}gb

# Time limit days-hrs:min:sec
#SBATCH --time {run_time}

# Standard output and error log
#SBATCH --output=slurm_out/{job_name}.out

module load fsl-6.0.3
conda activate fmri

{script_name}

"""
    return sbatch_setup

def create_job(job_name, job_cmd):
    print(job_name)
    f = open(f"{job_name}.sh", "a")
    f.writelines(setup_sbatch(job_name, job_cmd))
    f.close()

    subprocess.run(['sbatch', f"{job_name}.sh"],check=True, capture_output=True, text=True)
    os.remove(f"{job_name}.sh")

n = 0 
for sub in sub_list:
    sub_dir = f"{study_dir}/{sub}/ses-0{ses}"
    if preprocess == True:
        job_name = f'{sub}_preprocess'
        

        job_cmd = f'python preprocess.py {exp} {sub}'
        create_job(job_name, job_cmd)
        n+=1
        

    if run_1stlevel == True:
        for task in tasks:
            task_dir = f'{sub_dir}/derivatives/fsl/{task}'
            for run in runs:
                
                job_name = f'{sub}_{task}_{run}'
                job_cmd = f'feat {task_dir}/run-0{run}/1stLevel{suf}.fsf'

                #check if the feat file exists
                if os.path.exists(f'{task_dir}/run-0{run}/1stLevel{suf}.fsf'):

                    create_job(job_name, job_cmd)
                    n += 1

    if run_highlevel == True:
        for task in tasks:
            task_dir = f'{sub_dir}/derivatives/fsl/{task}'
        
            job_name = f'{sub}_{task}_high'
            job_cmd = f'feat {task_dir}/HighLevel{suf}.fsf'

            #check if the feat file exists
            if os.path.exists(f'{task_dir}/HighLevel{suf}.fsf'):

                create_job(job_name, job_cmd)
                n += 1



    if n >= pause_crit:
        #wait X minutes
        time.sleep(pause_time*60)
        n = 0
            






