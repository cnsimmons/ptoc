#this never worked - i kept clearing memory but it kept hitting out of memory errors
import subprocess
import os
import time

# Job parameters
job_name = 'fc_ppi'
mem = 120  # GB
run_time = "72:00:00"  # 72 hours

pause_crit = 3  # number of jobs to request before pausing
pause_time = 10  # minutes to wait between job batches

# Add import for params
import sys
sys.path.insert(0, '/user_data/csimmon2/git_repos/ptoc')
import ptoc_params as params

# Output directory for slurm logs
slurm_out_dir = os.path.join('/user_data/csimmon2/git_repos/ptoc', 'slurm_out')
os.makedirs(slurm_out_dir, exist_ok=True)

# Path to analysis script - full path
fc_script = '/user_data/csimmon2/git_repos/ptoc/tools/ppi_fc.py'

def setup_sbatch(job_name, script_path, sub):
    sbatch_setup = f"""#!/bin/bash -l

#SBATCH --job-name={job_name}_{sub}
#SBATCH --mail-type=ALL
#SBATCH --mail-user=csimmon2@andrew.cmu.edu
#SBATCH -p gpu
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem={mem}gb
#SBATCH --time={run_time}
#SBATCH --output={os.path.join(slurm_out_dir, f"{job_name}_{sub}.out")}

module load fsl-6.0.3
conda activate fmri

python {script_path} {sub}
"""
    return sbatch_setup

def create_job(job_name, script_path, sub):
    print(f"Creating job: {job_name}_{sub}")
    
    batch_filename = f"{job_name}_{sub}.sh"
    with open(batch_filename, "w") as f:
        f.write(setup_sbatch(job_name, script_path, sub))

    subprocess.run(['sbatch', batch_filename], check=True, capture_output=True, text=True)
    os.remove(batch_filename)

# Submit jobs
n = 0
sub_list = ['sub-spaceloc2013']  # Add your subjects here
for sub in sub_list:
    create_job(job_name, fc_script, sub)
    n += 1

    if n >= pause_crit:
        print(f"Pausing for {pause_time} minutes...")
        time.sleep(pause_time * 60)
        n = 0

print("All jobs submitted!")