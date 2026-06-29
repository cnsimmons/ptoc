#!/usr/bin/env python3
"""
Generate V1 sphere coordinates for Experiment 1 (loc) control ROI (R2.2).

V1 is NOT object-selective, so it is seeded from the scramble > object peak
(the negated object > scramble contrast, cope 3 / zstat3_anat) inside the
V1 parcel, hemisphere-split. This mirrors the peak-finding logic of
extract_roi_coords() in exp2_fc_ppi.py.

SAFE BY DESIGN:
- Reads the existing object-ROI coords only to copy the per-index/run-combo
  structure; never writes to sphere_coords_hemisphere.csv.
- Writes V1-only rows to a SEPARATE file (v1_coords.csv) for inspection.
- Merge into sphere_coords_hemisphere.csv is a deliberate, separate step
  (see --merge, off by default).

Schema matches sphere_coords_hemisphere.csv:
    index,task,roi,hemisphere,x,y,z
"""

import os
import sys
import argparse
import shutil
import pandas as pd
import numpy as np
from nilearn import image, plotting

curr_dir = '/user_data/csimmon2/git_repos/ptoc'
sys.path.insert(0, curr_dir)
import ptoc_params as params

raw_dir = params.raw_dir          # /lab_data/behrmannlab/vlad/hemispace
study_dir = '/lab_data/behrmannlab/vlad/ptoc'

# Exp 1 leave-one-out run combos, indexed 0,1,2 — matches existing coords.
RUN_COMBOS = [[1, 2], [1, 3], [2, 3]]
COPE = 3                          # Object > Scramble; negated below for scramble > object
PARCEL = 'V1'
TASK = 'loc'


def get_controls():
    sub_info = pd.read_csv(f'{curr_dir}/sub_info.csv')
    return sub_info[sub_info['group'] == 'control']['sub'].tolist()


def v1_coords_for_subject(ss):
    """Return list of coord rows for one subject, or [] if anything is missing."""
    rows = []
    roi_dir = f'{study_dir}/{ss}/ses-01/derivatives/rois'
    parcel_path = f'{roi_dir}/parcels/{PARCEL}.nii.gz'

    if not os.path.exists(parcel_path):
        print(f'[{ss}] V1 parcel not found: {parcel_path} — skipping')
        return rows

    roi = image.load_img(parcel_path)
    roi_data = roi.get_fdata()
    center_x = roi_data.shape[0] // 2

    for idx, rc in enumerate(RUN_COMBOS):
        # Load the two defining runs' zstat3, in anat space, and mean them.
        zstats = []
        for rn in rc:
            zpath = (f'{raw_dir}/{ss}/ses-01/derivatives/fsl/{TASK}/'
                     f'run-0{rn}/1stLevel.feat/stats/zstat{COPE}_anat.nii.gz')
            if not os.path.exists(zpath):
                print(f'[{ss}] missing {zpath} — skipping index {idx}')
                zstats = []
                break
            zstats.append(image.load_img(zpath))

        if not zstats:
            continue

        mean_z = image.mean_img(zstats) if len(zstats) > 1 else zstats[0]
        # scramble > object = negate object > scramble
        scram_gt_obj = image.math_img('-img', img=mean_z)

        for lr, sl in [('left', slice(None, center_x)),
                       ('right', slice(center_x, None))]:
            hemi_data = np.zeros_like(roi_data)
            hemi_data[sl] = roi_data[sl]
            if np.sum(hemi_data) == 0:
                print(f'[{ss}] empty {lr} V1 hemi — skipping')
                continue

            hemi_roi = image.new_img_like(roi, hemi_data)
            hemi_roi = image.math_img('img > 0', img=hemi_roi)

            coords = plotting.find_xyz_cut_coords(
                scram_gt_obj, mask_img=hemi_roi, activation_threshold=0.99)

            rows.append({
                'index': idx, 'task': TASK, 'roi': PARCEL, 'hemisphere': lr,
                'x': coords[0], 'y': coords[1], 'z': coords[2],
                'subject': ss,
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('subjects', nargs='*', help='subject ids (default: all controls)')
    ap.add_argument('--merge', action='store_true',
                    help='after generating, append V1 rows into each subject\'s '
                         'sphere_coords_hemisphere.csv (backs up original first)')
    args = ap.parse_args()

    subs = args.subjects if args.subjects else get_controls()

    all_rows = []
    for ss in subs:
        rows = v1_coords_for_subject(ss)
        all_rows.extend(rows)
        print(f'[{ss}] generated {len(rows)} V1 rows')

    if not all_rows:
        print('No V1 rows generated.')
        return

    df = pd.DataFrame(all_rows)
    out = f'{curr_dir}/tools/v1_coords.csv'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f'\nWrote {len(df)} rows to {out}')
    print('Inspect this before merging. Original coords untouched.')

    if args.merge:
        cols = ['index', 'task', 'roi', 'hemisphere', 'x', 'y', 'z']
        for ss in df['subject'].unique():
            target = (f'{study_dir}/{ss}/ses-01/derivatives/rois/'
                      f'spheres/sphere_coords_hemisphere.csv')
            if not os.path.exists(target):
                print(f'[{ss}] no existing coords file — skipping merge')
                continue
            existing = pd.read_csv(target)
            if ((existing['roi'] == PARCEL).any()):
                print(f'[{ss}] V1 already in coords — skipping (no overwrite)')
                continue
            shutil.copy(target, target + '.bak')
            v1 = df[df['subject'] == ss][cols]
            merged = pd.concat([existing, v1], ignore_index=True)
            merged.to_csv(target, index=False)
            print(f'[{ss}] merged {len(v1)} V1 rows (backup: {target}.bak)')


if __name__ == '__main__':
    main()