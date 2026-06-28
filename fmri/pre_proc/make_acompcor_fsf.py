"""
Create aCompCor FEAT design files — ONE subject, all runs.

For each run, copies the source 1stLevel.fsf to 1stLevel_acompcor.fsf and
changes ONLY two lines:
  - fmri(outputdir)      -> .../1stLevel_acompcor   (separate output dir)
  - confoundev_files(1)  -> the combined aCompCor+spikes confound file

Everything else in the design is left identical. Original .fsf and .feat
are not touched. Does NOT run FEAT.

Run:  python make_acompcor_fsf.py sub-083
"""

import os
import re
import sys

raw_dir = '/lab_data/behrmannlab/vlad/hemispace'
runs = [1, 2, 3]
suf = '_acompcor'

def main(ss):
    base = f'{raw_dir}/{ss}/ses-01/derivatives/fsl/loc'
    acomp_dir = f'{raw_dir}/{ss}/ses-01/derivatives/acompcor'

    for rn in runs:
        run_dir = f'{base}/run-0{rn}'
        src_fsf = f'{run_dir}/1stLevel.fsf'
        dst_fsf = f'{run_dir}/1stLevel{suf}.fsf'
        confound = f'{acomp_dir}/{ss}_run-0{rn}_confounds_combined.txt'

        if not os.path.exists(src_fsf):
            print(f'[run {rn}] source .fsf not found, skipping: {src_fsf}')
            continue
        if not os.path.exists(confound):
            print(f'[run {rn}] confound file not found, skipping: {confound}')
            continue

        with open(src_fsf) as f:
            lines = f.readlines()

        new_outputdir = f'{run_dir}/1stLevel{suf}'
        changed_out = changed_conf = False
        out = []
        for ln in lines:
            if re.match(r'\s*set fmri\(outputdir\)', ln):
                out.append(f'set fmri(outputdir) "{new_outputdir}"\n')
                changed_out = True
            elif re.match(r'\s*set confoundev_files\(1\)', ln):
                out.append(f'set confoundev_files(1) "{confound}"\n')
                changed_conf = True
            else:
                out.append(ln)

        if not changed_out:
            print(f'[run {rn}] WARNING: outputdir line not found')
        if not changed_conf:
            print(f'[run {rn}] WARNING: confoundev_files(1) line not found')

        with open(dst_fsf, 'w') as f:
            f.writelines(out)
        print(f'[run {rn}] wrote {dst_fsf}')
        print(f'           outputdir -> {new_outputdir}')
        print(f'           confounds -> {confound}')

    print('\nDone. aCompCor .fsf files written. FEAT NOT run.')

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python make_acompcor_fsf.py <subject>')
        sys.exit(1)
    main(sys.argv[1])