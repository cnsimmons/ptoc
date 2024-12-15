import subprocess
import os
import time
import pandas as pd

# Job parameters
job_name = 'gca_searchlight'
mem = 120  # GB
run_time = "72:00:00"  # 72 hours

pause_crit = 3  # number of jobs to request before pausing
pause_time = 10  # how long to wait between job batches in minutes

exp = 'ptoc'
tasks = ['toolloc']

# Project directories
project_dir = '/user_data/csimmon2/git_repos/ptoc'
study_dir = f'/lab_data/behrmannlab/vlad/{exp}'
slurm_out_dir = os.path.join(project_dir, 'slurm_out')

# Ensure slurm_out directory exists
os.makedirs(slurm_out_dir, exist_ok=True)

# Load and filter subjects
sub_info = pd.read_csv(f'{project_dir}/sub_info_tool.csv')
#sub_list = sub_info[sub_info['exp'] == 'spaceloc']['sub'].tolist()

#sub_list = sub_info[
    #(sub_info['exp'] == 'spaceloc') & 
    #(sub_info['sub'] != 'sub-spaceloc1001')
#]['sub'].tolist()

sub_list = ['sub-spaceloc2013', 'sub-spaceloc2014', 'sub-spaceloc2017', 'sub-spaceloc2018']  # Test

print(f"Processing subjects: {sub_list}")

# Path to your GCA analysis script
gca_script = os.path.join(project_dir, 'tools', 'gca_searchlight_tools.py')

def setup_sbatch(job_name, script_name, sub):
    sbatch_setup = f"""#!/bin/bash -l

# Job name
#SBATCH --job-name={job_name}_{sub}

#SBATCH --mail-type=ALL
#SBATCH --mail-user=csimmon2@andrew.cmu.edu

# Submit job to gpu queue                
#SBATCH -p gpu

#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem={mem}gb

# Time limit hrs:min:sec
#SBATCH --time {run_time}

# Standard output and error log
#SBATCH --output={os.path.join(slurm_out_dir, f"{job_name}_{sub}.out")}

module load fsl-6.0.3
conda activate brainiak_env

# Check Brainiak GPU configuration
python -c "
import brainiak.utils.fmriroi
import brainiak.searchlight.searchlight
print('Brainiak searchlight backend:', brainiak.searchlight.searchlight.Searchlight._backend)
"

{script_name} {sub}

"""
    return sbatch_setup

def create_job(job_name, job_cmd, sub):
    print(f"Creating job: {job_name}_{sub}")
    with open(f"{job_name}_{sub}.sh", "w") as f:
        f.write(setup_sbatch(job_name, job_cmd, sub))

    subprocess.run(['sbatch', f"{job_name}_{sub}.sh"], check=True, capture_output=True, text=True)
    os.remove(f"{job_name}_{sub}.sh")

n = 0
for sub in sub_list:
    # Run the GCA script with subject as argument
    job_cmd = f'python {gca_script}'
    
    create_job(job_name, job_cmd, sub)
    n += 1

    if n >= pause_crit:
        print(f"Pausing for {pause_time} minutes...")
        time.sleep(pause_time * 60)
        n = 0

print("All jobs submitted!")