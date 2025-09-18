import subprocess
import os
import time

# --- Job Parameters ---
job_name = 'partial_residual'
mem = 128 # Memory in GB
run_time = "0-04:00:00" # 4 hours for a single subject
pause_crit = 12 # Submit a batch of 12 jobs at a time
pause_time = 5 # Wait 30 minutes between batches

# --- CORRECTED Subject List ---
# Extracted from the CSV data you provided. (24 subjects total)
sub_list = [
    'sub-025', 'sub-038', 'sub-057', 'sub-059', 'sub-064',
    'sub-067', 'sub-068', 'sub-071', 'sub-083', 'sub-084',
    'sub-085', 'sub-087', 'sub-088', 'sub-093', 'sub-094',
    'sub-095', 'sub-096', 'sub-097', 'sub-107'
]

# --- Path to your analysis script ---
# Make sure this path is correct
script_to_run = '/user_data/csimmon2/git_repos/ptoc/being_verified/exp_1/fc_ppi/batch_partial_residual.py'

def setup_sbatch(job_name, script_cmd):
    """Creates the content of the SLURM sbatch script."""
    sbatch_setup = f"""#!/bin/bash -l

# Job name
#SBATCH --job-name={job_name}

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=csimmon2@andrew.cmu.edu

# Job partition and resources
#SBATCH -p cpu
#SBATCH --mem={mem}gb
#SBATCH --time={run_time}

# Standard output and error log
#SBATCH --output=slurm_out/{job_name}.out
#SBATCH --error=slurm_out/{job_name}.err

# Activate conda environment and run the python script
conda activate fmri
{script_cmd}

"""
    return sbatch_setup

def create_job(job_name, job_cmd):
    """Writes the sbatch script, submits it, and then cleans up."""
    script_filename = f"{job_name}.sh"
    with open(script_filename, "w") as f:
        f.writelines(setup_sbatch(job_name, job_cmd))
    
    try:
        subprocess.run(['sbatch', script_filename], check=True, capture_output=True, text=True)
        print(f"✅ Submitted job: {job_name}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error submitting {job_name}:")
        print(e.stderr)
    finally:
        os.remove(script_filename)

# --- Main Loop ---
# Create the directory for SLURM output files if it doesn't exist
os.makedirs('slurm_out', exist_ok=True)

n_jobs = 0
print(f"Starting submission for {len(sub_list)} subjects.")
for sub in sub_list:
    # Define a unique name for this subject's job
    current_job_name = f'{sub}_partial_residual'
    
    # The command that will be executed in the sbatch script
    # This passes the subject ID to your analysis script
    job_command = f'python {script_to_run} {sub}'
    
    create_job(current_job_name, job_command)
    n_jobs += 1
    
    # Pause submission to be kind to the scheduler
    if n_jobs >= pause_crit and sub != sub_list[-1]:
        print(f"--- Pausing for {pause_time} minutes ---")
        time.sleep(pause_time * 60)
        n_jobs = 0

print("\nAll jobs submitted!")