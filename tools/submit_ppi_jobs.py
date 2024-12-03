import subprocess
from glob import glob
import os
import time
import pandas as pd

# SLURM configuration
job_name = 'ppi_job'
mem = "120GB"
run_time = "1-00:00:00"
pause_crit = 4  # number of jobs to request
pause_time = 40  # how long to wait between jobs in minutes

# Define spaceloc subjects
subjects = [f'sub-spaceloc{i:04d}' for i in range(1007, 1013)]  # 1003-1012, 1 and 2 are already complete, 3, 4, 5, and 6 are running
subjects.extend([f'sub-spaceloc{i}' for i in range(2013, 2018)])  # 2013-2018, 2018 is running

# Remove sub-spaceloc1001 as it's already complete
#subjects.remove('sub-spaceloc1001')

# Define ROIs to process
rois = ['LO', 'pIPS']  # You can add more ROIs to this list as needed

def setup_sbatch(job_name, subject, rois):
    rois_str = ' '.join(rois)  # Convert ROI list to space-separated string
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
python tools/ppi_fc.py {subject} --rois {rois_str}
"""
    return sbatch_setup

def create_job(job_name, subject, rois):
    print(f"Creating job for {subject} with ROIs: {', '.join(rois)}")
    with open(f"{job_name}.sh", "w") as f:
        f.write(setup_sbatch(job_name, subject, rois))
    subprocess.run(['sbatch', f"{job_name}.sh"], check=True, capture_output=True, text=True)
    os.remove(f"{job_name}.sh")

if __name__ == "__main__":
    print(f"Processing {len(subjects)} subjects with ROIs: {', '.join(rois)}")
    print(subjects)
    
    # Create output directory for SLURM logs
    os.makedirs('slurm_out', exist_ok=True)
    
    # Submit jobs
    n = 0
    for sub in subjects:
        job_name = f'ppi_{sub}'
        create_job(job_name, sub, rois)
        n += 1
        if n >= pause_crit:
            print(f"Reached {pause_crit} jobs, waiting {pause_time} minutes...")
            time.sleep(pause_time * 60)
            n = 0