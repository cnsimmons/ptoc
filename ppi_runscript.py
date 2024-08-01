import subprocess
import os
import time
import pandas as pd

# Job configuration
job_name = 'ppi_job'
mem = 32
run_time = "1-00:00:00"

pause_crit = 12  # Number of jobs to request before pausing
pause_time = 40  # How long to wait between jobs in minutes

# List of runs and tasks
runs = list(range(1, 4))
exp = 'ptoc'
tasks = ['loc']

# Load subject information
# sub_info = pd.read_csv('/user_data/csimmon2/git_repos/ptoc/sub_info.csv')
# sub_list = sub_info['sub'].tolist()

# Specify the subject list to process
sub_list = ['sub-038']

print(sub_list)

study_dir = f'/lab_data/behrmannlab/vlad/{exp}'
ses = 1
suf = ''

# Function to set up sbatch script
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
#SBATCH --mem={mem}gb
#SBATCH --time={run_time}
#SBATCH --output=slurm_out/{job_name}.out

module load fsl-6.0.3
conda activate fmri

{script_name}
"""
    return sbatch_setup

# Function to create and submit a job
def create_job(job_name, job_cmd):
    print(job_name)
    with open(f"{job_name}.sh", "w") as f:
        f.writelines(setup_sbatch(job_name, job_cmd))
    subprocess.run(['sbatch', f"{job_name}.sh"], check=True, capture_output=True, text=True)
    os.remove(f"{job_name}.sh")

# Loop through subjects and ROIs to create and submit jobs
n = 0
rois = ['LO']  # List of ROIs

for sub in sub_list:
    sub_dir = f"{study_dir}/{sub}/ses-0{ses}"
        
    # PPI analysis for each ROI
    for roi in rois:
        job_name = f'{sub}_ppi_{roi}'
        job_cmd = f'python ppi_analyses_slurm.py --sub {sub} --roi {roi}'
        create_job(job_name, job_cmd)
        n += 1

    if n >= pause_crit:
        time.sleep(pause_time * 60)
        n = 0