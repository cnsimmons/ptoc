cat > ~/run_fcppi.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=fcppi
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/csimmon2/fcppi_%j.log

module load fsl-6.0.3
source ~/anaconda3/etc/profile.d/conda.sh
conda activate fmri
cd /user_data/csimmon2/git_repos/ptoc
python being_verified/exp_1/fc_ppi/compcor_fc_ppi_arg.py sub-083 sub-093 sub-107
python being_verified/exp_1/fc_ppi/compcor_fc_ppi_arg.py sub-057 sub-059 sub-064 sub-067 sub-068 sub-071 sub-084 sub-085 sub-087 sub-088 sub-094 sub-095 sub-096 sub-097
EOF

sbatch ~/run_fcppi.sh