"""
Register aCompCor 1stLevel output to anat — ONE subject, all runs.

Mirrors register_1stlevel.py but targets the 1stLevel_acompcor.feat dir and
writes filtered_func_data_reg / zstat_reg there, so the original (non-aCompCor)
_reg files are never overwritten.

Run:  python register_1stlevel_acompcor.py sub-083
      python register_1stlevel_acompcor.py --all-controls [--force]
"""

import os
import sys
import subprocess

curr_dir = '/user_data/csimmon2/git_repos/ptoc'
sys.path.append(curr_dir)
import ptoc_params as params

raw_dir = params.raw_dir
data_dir = params.data_dir
cope = 3                   # object>scramble; only used to register a zstat (optional)

suf = '_acompcor'          # targets 1stLevel_acompcor.feat
task = 'loc'
runs = [1, 2, 3]

def main(ss, force=False):
    sub_data = f'{raw_dir}/{ss}/ses-01'
    anat = f'{raw_dir}/{ss}/ses-01/anat/{ss}_ses-01_T1w_brain.nii.gz'

    for rn in runs:
        run_dir = f'{sub_data}/derivatives/fsl/{task}/run-0{rn}/1stLevel{suf}.feat'
        filtered_func = f'{run_dir}/filtered_func_data.nii.gz'
        out_func = f'{run_dir}/filtered_func_data_reg.nii.gz'
        xfm = f'{run_dir}/reg/example_func2standard.mat'

        zstat_func = f'{run_dir}/stats/zstat{cope}.nii.gz'
        zstat_out = f'{run_dir}/stats/zstat{cope}_reg.nii.gz'

        if os.path.exists(out_func) and not force:
            print(f'[run {rn}] reg output exists, skipping (use --force): {out_func}')
            continue
        if not os.path.exists(filtered_func):
            print(f'[run {rn}] no filtered_func, skipping: {filtered_func}')
            continue
        if not os.path.exists(xfm):
            print(f'[run {rn}] no reg matrix, skipping: {xfm}')
            continue

        print(f'[run {rn}] registering filtered_func -> anat')
        subprocess.run(
            f'flirt -in {filtered_func} -ref {anat} -out {out_func} '
            f'-applyxfm -init {xfm} -interp trilinear'.split(), check=True)

        if os.path.exists(zstat_func):
            print(f'[run {rn}] registering zstat{cope} -> anat')
            subprocess.run(
                f'flirt -in {zstat_func} -ref {anat} -out {zstat_out} '
                f'-applyxfm -init {xfm} -interp trilinear'.split(), check=True)
        else:
            print(f'[run {rn}] zstat{cope} not found (ok if not needed)')

        print(f'[run {rn}] done -> {out_func}')

    print('\nDone. aCompCor outputs registered to anat.')

if __name__ == '__main__':
    import pandas as pd
    args = sys.argv[1:]
    force = '--force' in args
    args = [a for a in args if a != '--force']
    if len(args) != 1:
        print('Usage: python register_1stlevel_acompcor.py <subject|--all-controls> [--force]')
        sys.exit(1)
    arg = args[0]
    if arg == '--all-controls':
        info = pd.read_csv('/user_data/csimmon2/git_repos/ptoc/sub_info.csv')
        subs = info[info['group'] == 'control']['sub'].tolist()
        subs = [s if str(s).startswith('sub-') else f'sub-{s}' for s in subs]
        for ss in subs:
            print(f'\n########## {ss} ##########')
            try:
                main(ss, force=force)
            except Exception as e:
                print(f'ERROR on {ss}: {e}')
    else:
        main(arg, force=force)