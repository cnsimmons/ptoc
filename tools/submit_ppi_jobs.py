import subprocess
from glob import glob
import os
import time
import pandas as pd

# SLURM configuration
job_name = 'ppi_job'
mem = "120GB"
run_time = "1-00:00:00"
pause_crit = 6  # number of jobs to request
pause_time = 40  # how long to wait between jobs in minutes

# Define spaceloc subjects
subjects = [f'sub-spaceloc{i:04d}' for i in range(1003, 1013)]  # 1003-1012, 1 and 2 are already complete
subjects.extend([f'sub-spaceloc{i}' for i in range(2013, 2019)])  # 2013-2018
# Remove sub-spaceloc1001 as it's already complete
#subjects.remove('sub-spaceloc1001')

def setup_sbatch(job_name, subject):
    """Create SBATCH script content."""
    sbatch_setup = f"""#!/bin/bash -l
#SBATCH --job-name={job_name}
#SBATCH --mail-type=ALL
#SBATCH --mail-user=csimmon2@andrew.cmu.edu
#SBATCH -p gpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem={mem}
#SBATCH --time={run_time}
#SBATCH --output=slurm_out/{job_name}.out

module load fsl-6.0.3
conda activate fmri
python tools/ppi_fc.py {subject}
"""
    return sbatch_setup

def create_job(job_name, subject):
    """Create and submit SLURM job."""
    print(f"Creating job for {subject}")
    with open(f"{job_name}.sh", "w") as f:
        f.write(setup_sbatch(job_name, subject))
    subprocess.run(['sbatch', f"{job_name}.sh"], check=True, capture_output=True, text=True)
    os.remove(f"{job_name}.sh")

if __name__ == "__main__":
    print(f"Processing {len(subjects)} subjects:")
    print(subjects)
    
    # Create output directory for SLURM logs
    os.makedirs('slurm_out', exist_ok=True)
    
    # Submit jobs
    n = 0
    for sub in subjects:
        job_name = f'ppi_{sub}'
        create_job(job_name, sub)
        n += 1
        if n >= pause_crit:
            print(f"Reached {pause_crit} jobs, waiting {pause_time} minutes...")
            time.sleep(pause_time * 60)
            n = 0