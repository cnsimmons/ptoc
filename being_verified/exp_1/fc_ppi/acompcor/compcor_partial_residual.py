#!/usr/bin/env python3
"""
aCompCor partial correlation analysis for Experiment 1.

For each hemisphere, computes:
  - pIPS → brain connectivity after regressing out LO
  - LO → brain connectivity after regressing out pIPS

Reads from 1stLevel_acompcor.feat directories.
Residualized seed is z-scored → output is true partial Pearson's r.
Uses sequential run loading to keep peak memory ~35 GB.

Usage:
    python compcor_partial_residual.py sub-025 sub-038 sub-083
    python compcor_partial_residual.py sub-025 --force
"""

import os
import sys
import gc
import time
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import image, input_data
from nilearn.maskers import NiftiMasker

# ── paths ──────────────────────────────────────────────────────────────────────
curr_dir = '/user_data/csimmon2/git_repos/ptoc'
sys.path.insert(0, curr_dir)
import ptoc_params as params

raw_dir   = params.raw_dir                        # /lab_data/behrmannlab/vlad/hemispace
study_dir = '/lab_data/behrmannlab/vlad/ptoc'

# ── analysis parameters ────────────────────────────────────────────────────────
hemispheres = ['left', 'right']
run_combos  = [[1, 2], [1, 3], [2, 3]]

# Each entry: (seed_roi, control_roi)
# pIPS cleaned of LO  → dorsal-specific map
# LO cleaned of pIPS  → ventral-specific map
partial_pairs = [
    ('pIPS', 'LO'),
    ('LO',   'pIPS'),
]


# ── helper functions ───────────────────────────────────────────────────────────

def extract_roi_sphere(img, coords, radius=6):
    """Extract mean timeseries from 6 mm sphere. Returns 1-D array (N,)."""
    masker = input_data.NiftiSpheresMasker([tuple(coords)], radius=radius)
    ts = masker.fit_transform(img)
    return np.mean(ts, axis=1)


# ── main analysis ──────────────────────────────────────────────────────────────

def conduct_analyses(subs, force=False):
    for ss in subs:
        print(f'\n=== {ss} ===')

        sub_dir  = f'{study_dir}/{ss}/ses-01/derivatives'
        temp_dir = f'{raw_dir}/{ss}/ses-01/derivatives/fsl/loc'
        roi_dir  = f'{study_dir}/{ss}/ses-01/derivatives/rois'

        # ── coordinates ────────────────────────────────────────────────────
        coord_file = f'{roi_dir}/spheres/sphere_coords_hemisphere.csv'
        if not os.path.exists(coord_file):
            print('  missing coordinate file — skipping subject')
            continue
        roi_coords = pd.read_csv(coord_file)

        # ── brain mask ─────────────────────────────────────────────────────
        mask_path = f'{raw_dir}/{ss}/ses-01/anat/{ss}_ses-01_T1w_brain_mask.nii.gz'
        if not os.path.exists(mask_path):
            print('  missing brain mask — skipping subject')
            continue
        whole_brain_mask = nib.load(mask_path)
        brain_masker = NiftiMasker(whole_brain_mask, smoothing_fwhm=0,
                                   standardize=True)

        os.makedirs(f'{sub_dir}/fc', exist_ok=True)

        # ── loop over partial pairs × hemispheres ──────────────────────────
        for seed_roi, control_roi in partial_pairs:
            for hemi in hemispheres:
                t0 = time.time()

                out_file = (f'{sub_dir}/fc/{ss}_{seed_roi}_clean_'
                            f'{hemi}_loc_fc_acompcor.nii.gz')

                if not force and os.path.exists(out_file):
                    print(f'  {seed_roi} (ctrl {control_roi}) {hemi}: '
                          f'exists — skipping')
                    continue

                print(f'  {seed_roi} (ctrl {control_roi}) {hemi} ...',
                      end=' ', flush=True)

                all_runs = []

                for rcn, rc in enumerate(run_combos):

                    # ── look up coordinates for both ROIs ──────────────────
                    def get_coords(roi_name):
                        cc = roi_coords[
                            (roi_coords['index'] == rcn) &
                            (roi_coords['task'] == 'loc') &
                            (roi_coords['roi'] == roi_name) &
                            (roi_coords['hemisphere'] == hemi)
                        ]
                        if cc.empty:
                            return None
                        return cc[['x', 'y', 'z']].values.tolist()[0]

                    seed_xyz    = get_coords(seed_roi)
                    control_xyz = get_coords(control_roi)

                    if seed_xyz is None or control_xyz is None:
                        continue

                    # ── sequential loading: one run at a time ──────────────
                    run_seed    = []
                    run_control = []
                    run_brain   = []
                    valid = True

                    for rn in rc:
                        fpath = (f'{temp_dir}/run-0{rn}/'
                                 f'1stLevel_acompcor.feat/'
                                 f'filtered_func_data_reg.nii.gz')
                        if not os.path.exists(fpath):
                            print(f'\n    missing {fpath}')
                            valid = False
                            break

                        img = image.clean_img(image.load_img(fpath),
                                              standardize=True)

                        run_seed.append(extract_roi_sphere(img, seed_xyz))
                        run_control.append(
                            extract_roi_sphere(img, control_xyz))
                        run_brain.append(brain_masker.fit_transform(img))

                        del img
                        gc.collect()

                    if not valid:
                        continue

                    # ── concatenate across runs ────────────────────────────
                    seed_ts    = np.concatenate(run_seed)
                    control_ts = np.concatenate(run_control)
                    brain_ts   = np.concatenate(run_brain, axis=0)
                    del run_seed, run_control, run_brain
                    gc.collect()

                    # trim to common length
                    n = min(len(seed_ts), len(control_ts), brain_ts.shape[0])
                    seed_ts    = seed_ts[:n]
                    control_ts = control_ts[:n]
                    brain_ts   = brain_ts[:n]

                    # ── residualize: remove control signal from seed ───────
                    beta = (np.dot(control_ts, seed_ts) /
                            np.dot(control_ts, control_ts))
                    seed_clean = seed_ts - beta * control_ts

                    # ── standardize residual → true partial Pearson's r ────
                    seed_clean = ((seed_clean - seed_clean.mean()) /
                                  seed_clean.std()).reshape(-1, 1)

                    # ── correlate with brain ───────────────────────────────
                    corr = np.dot(brain_ts.T, seed_clean) / n
                    corr = np.arctanh(corr.ravel())
                    all_runs.append(brain_masker.inverse_transform(corr))

                    del seed_ts, control_ts, brain_ts, seed_clean, corr
                    gc.collect()

                # ── save mean across run combos ────────────────────────────
                if all_runs:
                    nib.save(image.mean_img(all_runs), out_file)

                del all_runs
                gc.collect()

                print(f'{time.time() - t0:.0f}s')


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='aCompCor partial correlation for Experiment 1')
    parser.add_argument('subjects', nargs='+',
                        help='Subject IDs (e.g. sub-025 sub-038)')
    parser.add_argument('--force', action='store_true', default=False,
                        help='Overwrite existing output files')
    args = parser.parse_args()

    conduct_analyses(args.subjects, force=args.force)