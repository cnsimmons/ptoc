"""
Register aCompCor 1stLevel zstat to anat (and MNI), mirroring
1stLevel_zstat2anat_mni.py exactly — same transform matrices, same outputs —
but targeting 1stLevel_acompcor.feat instead of 1stLevel.feat.
"""
curr_dir = '/user_data/csimmon2/git_repos/ptoc'
import pandas as pd
import subprocess
import os
import sys
sys.path.append(curr_dir)
import ptoc_params as params

data_dir = '/lab_data/behrmannlab/vlad/hemispace'
task = 'loc'
runs = [1, 2, 3]
cope = 3
feat = '1stLevel_acompcor.feat'
mni = '/opt/fsl/6.0.3/data/standard/MNI152_T1_2mm_brain.nii.gz'


def register(sub):
    sub_dir = f'{data_dir}/{sub}/ses-01'
    for run in runs:
        print(f'Registering {sub} run-0{run} to anat and then to MNI')
        zstat = f'{sub_dir}/derivatives/fsl/{task}/run-0{run}/{feat}/stats/zstat{cope}.nii.gz'
        out_anat = f'{sub_dir}/derivatives/fsl/{task}/run-0{run}/{feat}/stats/zstat{cope}_anat.nii.gz'
        out_mni = f'{sub_dir}/derivatives/fsl/{task}/run-0{run}/{feat}/stats/zstat{cope}_mni.nii.gz'
        anat_ref = f'{sub_dir}/anat/{sub}_ses-01_T1w_brain.nii.gz'
        reg_dir = f'{sub_dir}/derivatives/fsl/{task}/run-0{run}/{feat}/reg'
        if os.path.exists(zstat):
            bash_cmd1 = (f'flirt -in {zstat} -ref {anat_ref} -out {out_anat} '
                         f'-applyxfm -init {reg_dir}/example_func2highres.mat -interp trilinear')
            subprocess.run(bash_cmd1.split(), check=True)
            bash_cmd2 = (f'flirt -in {out_anat} -ref {mni} -out {out_mni} '
                         f'-applyxfm -init {sub_dir}/anat/anat2stand.mat -interp trilinear')
            subprocess.run(bash_cmd2.split(), check=True)
            print(f'Completed registration for {sub} run-0{run}')
        else:
            print(f'zstat {zstat} does not exist for {sub} run-0{run}')


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 1:
        print('Usage: python 1stLevel_zstat2anat_mni_acompcor.py <subject|--all-controls>')
        sys.exit(1)
    if args[0] == '--all-controls':
        info = pd.read_csv(f'{curr_dir}/sub_info.csv')
        subs = info[info['group'] == 'control']['sub'].tolist()
        for s in subs:
            register(s)
    else:
        register(args[0])
