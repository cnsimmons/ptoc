#!/usr/bin/env python3
"""
aCompCor FC and PPI analysis for Experiment 1.

Reads from 1stLevel_acompcor.feat directories (aCompCor-preprocessed FEAT outputs).
Uses sequential run loading to keep peak memory ~35 GB.

FC branch:  seed is z-scored before correlation → output is true Pearson's r.
PPI branch: seed is NOT z-scored → matches existing aCompCor PPI outputs.

Usage:
    # FC only (default)
    python compcor_fc_ppi_arg.py sub-025 sub-038 sub-083

    # FC + PPI
    python compcor_fc_ppi_arg.py sub-025 sub-038 --ppi

    # PPI only
    python compcor_fc_ppi_arg.py sub-025 sub-038 --ppi --no-fc

    # Force overwrite existing files
    python compcor_fc_ppi_arg.py sub-025 --force
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
from nilearn.glm.first_level import compute_regressor

# ── paths ──────────────────────────────────────────────────────────────────────
curr_dir = '/user_data/csimmon2/git_repos/ptoc'
sys.path.insert(0, curr_dir)
import ptoc_params as params

raw_dir = params.raw_dir                          # /lab_data/behrmannlab/vlad/hemispace
study_dir = '/lab_data/behrmannlab/vlad/ptoc'

# ── analysis parameters ────────────────────────────────────────────────────────
rois = ['LO', 'pIPS', 'PFS', 'V1']
hemispheres = ['left', 'right']
run_num = 3
run_combos = [[1, 2], [1, 3], [2, 3]]
vols_per_run = 184
tr = 2.0


# ── helper functions ───────────────────────────────────────────────────────────

def extract_roi_sphere(img, coords):
    """Extract mean timeseries from 6 mm sphere. Returns (N, 1) array."""
    masker = input_data.NiftiSpheresMasker([tuple(coords)], radius=6)
    ts = masker.fit_transform(img)
    return np.mean(ts, axis=1).reshape(-1, 1)


def make_psy_cov(runs, ss):
    """Object (+1) vs scramble (−1) psychological covariate.

    Matches the original exp1_fc_ppi.py behaviour exactly:
    single-run time axis, onsets from both runs overlaid.
    """
    cov_dir = f'{raw_dir}/{ss}/ses-01/covs'
    times = np.arange(0, vols_per_run * tr, tr)          # single-run length
    full_cov = pd.DataFrame(columns=['onset', 'duration', 'value'])

    for rn in runs:
        ss_num = ss.split('-')[1]
        obj_file = f'{cov_dir}/catloc_{ss_num}_run-0{rn}_Object.txt'
        scr_file = f'{cov_dir}/catloc_{ss_num}_run-0{rn}_Scramble.txt'

        if not os.path.exists(obj_file) or not os.path.exists(scr_file):
            print(f'    covariate file missing for run {rn}')
            continue

        obj_cov = pd.read_csv(obj_file, sep='\t', header=None,
                              names=['onset', 'duration', 'value'])
        scr_cov = pd.read_csv(scr_file, sep='\t', header=None,
                              names=['onset', 'duration', 'value'])
        scr_cov['value'] *= -1
        full_cov = pd.concat([full_cov, obj_cov, scr_cov])

    full_cov = full_cov.sort_values(by='onset').reset_index(drop=True)
    cov = full_cov.to_numpy()
    cov = cov[cov[:, 0] < times[-1]]                     # drop onsets past end

    if cov.shape[0] == 0:
        print('    no valid covariate data — returning zeros')
        return np.zeros((vols_per_run, 1))

    psy, _ = compute_regressor(cov.T, 'spm', times)
    return psy


# ── main analysis ──────────────────────────────────────────────────────────────

def conduct_analyses(subs, do_fc=True, do_ppi=False, force=False):
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

        # ── output dirs ────────────────────────────────────────────────────
        if do_fc:
            os.makedirs(f'{sub_dir}/fc', exist_ok=True)
        if do_ppi:
            os.makedirs(f'{sub_dir}/ppi', exist_ok=True)

        # ── loop over ROIs × hemispheres ───────────────────────────────────
        for rr in rois:
            for hemi in hemispheres:
                t0 = time.time()

                fc_file  = f'{sub_dir}/fc/{ss}_{rr}_{hemi}_loc_fc_acompcor.nii.gz'
                ppi_file = f'{sub_dir}/ppi/{ss}_{rr}_{hemi}_loc_ppi_acompcor.nii.gz'

                run_fc  = do_fc  and (force or not os.path.exists(fc_file))
                run_ppi = do_ppi and (force or not os.path.exists(ppi_file))

                if not run_fc and not run_ppi:
                    print(f'  {rr} {hemi}: exists — skipping')
                    continue

                tags = []
                if run_fc:  tags.append('FC')
                if run_ppi: tags.append('PPI')
                print(f'  {rr} {hemi} [{"+".join(tags)}] ...', end=' ',
                      flush=True)

                all_fc  = []
                all_ppi = []

                for rcn, rc in enumerate(run_combos):

                    # ── look up coordinates ────────────────────────────────
                    cc = roi_coords[
                        (roi_coords['index'] == rcn) &
                        (roi_coords['task'] == 'loc') &
                        (roi_coords['roi'] == rr) &
                        (roi_coords['hemisphere'] == hemi)
                    ]
                    if cc.empty:
                        continue
                    coords = cc[['x', 'y', 'z']].values.tolist()[0]

                    # ── sequential loading: one run at a time ──────────────
                    run_phys  = []
                    run_brain = []
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

                        run_phys.append(extract_roi_sphere(img, coords))
                        run_brain.append(brain_masker.fit_transform(img))

                        del img
                        gc.collect()

                    if not valid:
                        continue

                    # ── concatenate across runs ────────────────────────────
                    phys     = np.concatenate(run_phys,  axis=0)
                    brain_ts = np.concatenate(run_brain, axis=0)
                    del run_phys, run_brain
                    gc.collect()

                    # trim to shorter if mismatch
                    n = min(phys.shape[0], brain_ts.shape[0])
                    phys     = phys[:n]
                    brain_ts = brain_ts[:n]

                    # ── FC: standardised seed → true Pearson's r ───────────
                    if run_fc:
                        phys_z = (phys - phys.mean()) / phys.std()
                        corr = np.dot(brain_ts.T, phys_z) / n
                        corr = np.arctanh(corr.ravel())
                        all_fc.append(brain_masker.inverse_transform(corr))
                        del phys_z, corr

                    # ── PPI: unstandardised seed (matches existing outputs) ─
                    if run_ppi:
                        psy = make_psy_cov(rc, ss)
                        n_ppi = min(phys.shape[0], psy.shape[0],
                                    brain_ts.shape[0])
                        ppi_reg = phys[:n_ppi] * psy[:n_ppi]
                        bt_ppi  = brain_ts[:n_ppi]
                        corr = np.dot(bt_ppi.T, ppi_reg) / n_ppi
                        corr = np.arctanh(corr.ravel())
                        all_ppi.append(brain_masker.inverse_transform(corr))
                        del psy, ppi_reg, bt_ppi, corr

                    del phys, brain_ts
                    gc.collect()

                # ── save ───────────────────────────────────────────────────
                if run_fc and all_fc:
                    nib.save(image.mean_img(all_fc), fc_file)
                if run_ppi and all_ppi:
                    nib.save(image.mean_img(all_ppi), ppi_file)

                del all_fc, all_ppi
                gc.collect()

                print(f'{time.time() - t0:.0f}s')


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='aCompCor FC / PPI for Experiment 1')
    parser.add_argument('subjects', nargs='+',
                        help='Subject IDs (e.g. sub-025 sub-038)')
    parser.add_argument('--ppi', action='store_true', default=False,
                        help='Also run PPI (default: FC only)')
    parser.add_argument('--no-fc', action='store_true', default=False,
                        help='Skip FC (use with --ppi for PPI-only)')
    parser.add_argument('--force', action='store_true', default=False,
                        help='Overwrite existing output files')
    args = parser.parse_args()

    conduct_analyses(args.subjects,
                     do_fc  = not args.no_fc,
                     do_ppi = args.ppi,
                     force  = args.force)