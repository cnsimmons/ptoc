import os
import sys
import subprocess
import ptoc_params as params

curr_dir = '/user_data/csimmon2/git_repos/ptoc'
sys.path.insert(0, curr_dir)

# Define the script to run
curr_script = f'{curr_dir}/fmri/pre_proc/register_1stlevel.py'

sub_info = params.sub_info

# Find the subjects that have not been processed yet
processed_subjects = [filename.split('_')[1] for filename in os.listdir(curr_dir) if filename.startswith('processed_')]
subjects_to_process = sub_info[~sub_info['sub'].isin(processed_subjects)]['sub']

# Run the script for each subject
for target_subject in subjects_to_process:
    try:
        bash_cmd = f'python {curr_script} {target_subject}'
        subprocess.run(bash_cmd.split(), check=True)
        # Mark the subject as processed
        open(os.path.join(curr_dir, f'processed_{target_subject}.txt'), 'w').close()
    except Exception as e:
        print(f'Failed on {target_subject}: {e}')