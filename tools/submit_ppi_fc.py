import subprocess
import sys
import os

# Job configuration
job_name = 'ppi_fc_analysis'
mem = 128
run_time = "1-00:00:00"
study_dir = "/user_data/csimmon2/git_repos/ptoc"
subject = sys.argv[1]

def setup_sbatch(job_name, script_name):
    return f"""#!/bin/bash -l
#SBATCH --job-name={job_name}
#SBATCH --mail-type=ALL
#SBATCH --mail-user=csimmon2@andrew.cmu.edu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128gb
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_out/{job_name}.out

module load fsl-6.0.3
conda activate fmri

python {os.path.join(study_dir, 'tools/ppi_fc.py')} {subject}
"""

def main():
    # Set paths
    os.makedirs("slurm_out", exist_ok=True)
    
    # Get subject ID from command line
    if len(sys.argv) > 1:
        subject = sys.argv[1]
        # Create batch script
        script_content = setup_sbatch(job_name, f'python {os.path.join(study_dir, "tools/ppi_fc.py")} {subject}')
        script_path = f"{job_name}.sh"
        
        with open(script_path, "w") as f:
            f.write(script_content)
        
        # Submit job
        subprocess.run(['sbatch', script_path], check=True)
        os.remove(script_path)
    else:
        print("Please provide subject ID as command line argument")

if __name__ == "__main__":
    main()