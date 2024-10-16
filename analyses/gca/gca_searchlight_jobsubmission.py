import subprocess
import os
import time
import pandas as pd
import shutil

# Job parameters
job_name = 'gca_searchlight'
mem = 120  # GB
run_time = "48:00:00"  # 48 hours

pause_crit = 2  # number of jobs to request before pausing (3 is conservative 5 is an option)
pause_time = 40  # how long to wait between job batches in minutes

exp = 'ptoc'
tasks = ['loc']

# Project directories
project_dir = '/user_data/csimmon2/git_repos/ptoc'
study_dir = f'/lab_data/behrmannlab/vlad/{exp}'
slurm_out_dir = os.path.join(project_dir, 'slurm_out')

# Ensure slurm_out directory exists
os.makedirs(slurm_out_dir, exist_ok=True)

# Load subject information
sub_info = pd.read_csv(os.path.join(project_dir, 'sub_info.csv'))
#sub_list = sub_info['sub'].tolist()

# For testing, you might want to use a smaller list:
#sub_list = ['sub-038']
sub_list = ['sub-038', 'sub-059']

print(f"Processing subjects: {sub_list}")

# Path to your GCA analysis script
gca_script = os.path.join(project_dir, 'analyses', 'gca', 'gca_searchlight.py')

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
#module load cuda  # Load CUDA module if available
conda activate brainiak_env

# Check Brainiak GPU configuration
python -c "
import brainiak.utils.fmriroi
import brainiak.searchlight.searchlight
print('Brainiak searchlight backend:', brainiak.searchlight.searchlight.Searchlight._backend)
#print('Brainiak CUDA available:', brainiak.utils.fmriroi.cuda_available())
"

{script_name}

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
    # Create a copy of the GCA script for this job
    job_specific_script = os.path.join(project_dir, f'gca_searchlight_{sub}.py')
    shutil.copy(gca_script, job_specific_script)
    
    # Modify the subs list in the copied script
    with open(job_specific_script, 'r') as file:
        filedata = file.read()
    filedata = filedata.replace("subs = ['sub-025']", f"subs = ['{sub}']")
    with open(job_specific_script, 'w') as file:
        file.write(filedata)
    
    # Run the modified GCA script
    job_cmd = f'python {job_specific_script}'
    
    create_job(job_name, job_cmd, sub)
    n += 1

    if n >= pause_crit:
        print(f"Pausing for {pause_time} minutes...")
        time.sleep(pause_time * 60)
        n = 0

    # Clean up the temporary script
    os.remove(job_specific_script)

print("All jobs submitted!")