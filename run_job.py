import subprocess
from glob import glob
import os
import time
import pdb
import pandas as pd

job_name = 'fsl_job'
mem = 24
run_time = "1-00:00:00"

pause_crit = 12 #number of jobs to request
pause_time = 40 #low long to wait between jobs in minutes

runs=list(range(1,4))
exp = 'ptoc'

tasks = ['loc']





sub_info = pd.read_csv('/user_data/csimmon2/git_repos/ptoc/sub_info.csv')
sub_list = sub_info['sub'].tolist()


sub_list = ['sub-057', 'sub-059']
print(sub_list)


study_dir= f'/lab_data/behrmannlab/vlad/{exp}'
ses = 1
suf = ''

#the sbatch setup info
run_1stlevel = False
run_highlevel = True

preprocess = False



def setup_sbatch(job_name, script_name):
    sbatch_setup = f"""#!/bin/bash -l

# Job name
#SBATCH --job-name={job_name}

#SBATCH --mail-type=ALL
#SBATCH --mail-user=csimmon2@andrew.cmu.edu

# Submit job to cpu queue                
#SBATCH -p cpu

#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0
# SBATCH --exclude=mind-0-15,mind-0-14,mind-0-16
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
            






