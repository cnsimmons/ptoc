import subprocess
import os
import time
import pandas as pd

# SLURM job parameters
job_name = 'register_1stlevel'
mem = 28
run_time = "1-00:00:00"
pause_crit = 12  # number of jobs to request before pausing
pause_time = 40  # how long to wait between job batches in minutes

# Study parameters
exp = 'ptoc'
tasks = ['loc']
runs = list(range(1, 4))
study_dir = f'/lab_data/behrmannlab/vlad/{exp}'
ses = 1
suf = ''

# Read subject list
sub_info = pd.read_csv('/user_data/csimmon2/git_repos/ptoc/sub_info.csv')
sub_list = sub_info['sub'].tolist()

# SLURM submission script template
def setup_sbatch(job_name, sub, task, run):
    return f"""#!/bin/bash -l
#SBATCH --job-name={job_name}_{sub}_{task}_run{run}
#SBATCH --mail-type=ALL
#SBATCH --mail-user=csimmon2@andrew.cmu.edu
#SBATCH -p cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem={mem}gb
#SBATCH --time={run_time}
#SBATCH --output=slurm_out/{job_name}_{sub}_{task}_run{run}.out

module load fsl-6.0.3

sub_dir="{study_dir}/{sub}/ses-0{ses}"
anat="${{sub_dir}}/anat/{sub}_ses-0{ses}_T1w_brain.nii.gz"
run_dir="${{sub_dir}}/derivatives/fsl/{task}/run-0{run}/1stLevel{suf}.feat"

# Register filtered func
flirt -in ${{run_dir}}/filtered_func_data.nii.gz -ref $anat -out ${{run_dir}}/filtered_func_data_reg.nii.gz -applyxfm -init ${{run_dir}}/reg/example_func2standard.mat -interp trilinear

# Register zstat
for zz in 1 2 3 4 5; do
    flirt -in ${{run_dir}}/stats/zstat${{zz}}.nii.gz -ref $anat -out ${{run_dir}}/stats/zstat${{zz}}_reg.nii.gz -applyxfm -init ${{run_dir}}/reg/example_func2standard.mat -interp trilinear
done
"""

def create_job(job_name, sub, task, run):
    script_content = setup_sbatch(job_name, sub, task, run)
    script_file = f"{job_name}_{sub}_{task}_run{run}.sh"
    
    with open(script_file, "w") as f:
        f.write(script_content)
    
    subprocess.run(['sbatch', script_file], check=True, capture_output=True, text=True)
    os.remove(script_file)

# Main job submission loop
n = 0
for sub in sub_list:
    for task in tasks:
        for run in runs:
            sub_dir = f"{study_dir}/{sub}/ses-0{ses}"
            run_dir = f'{sub_dir}/derivatives/fsl/{task}/run-0{run}/1stLevel{suf}.feat'
            
            # Check if the run directory exists before submitting the job
            if os.path.exists(run_dir):
                create_job(job_name, sub, task, run)
                n += 1
                
                if n >= pause_crit:
                    print(f"Submitted {n} jobs. Pausing for {pause_time} minutes.")
                    time.sleep(pause_time * 60)
                    n = 0
            else:
                print(f"Skipping {sub}, {task}, run-0{run} - directory not found")

print("All registration jobs submitted!")