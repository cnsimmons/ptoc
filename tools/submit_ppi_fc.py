import subprocess
import os

# Job configuration
job_name = 'ppi_fc_analysis'
mem = 32
run_time = "1-00:00:00"

def create_sbatch_script(study_dir):
    return f"""#!/bin/bash -l

#SBATCH --job-name={job_name}
#SBATCH --mail-type=ALL
#SBATCH --mail-user=csimmon2@andrew.cmu.edu
#SBATCH -p cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem={mem}gb
#SBATCH --time={run_time}
#SBATCH --output=slurm_out/{job_name}.out

module load fsl-6.0.3
conda activate fmri

python {os.path.join(study_dir, 'run_ppi_fc.py')}
"""

def main():
    # Set paths
    study_dir = "/user_data/csimmon2/git_repos/hemisphere"
    os.makedirs("slurm_out", exist_ok=True)
    
    # Create batch script
    script_content = create_sbatch_script(study_dir)
    script_path = f"{job_name}.sh"
    
    with open(script_path, "w") as f:
        f.write(script_content)
    
    # Submit job
    subprocess.run(['sbatch', script_path], check=True)
    os.remove(script_path)

if __name__ == "__main__":
    main()