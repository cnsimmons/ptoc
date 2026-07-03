"""
Dice comparison: original vs aCompCor PPI for 5 subjects.
Mirrors the notebook's analyze_dice_coefficients() but runs both pipelines 
side-by-side and prints a comparison table.

Requires: per-subject MNI maps from native2mni.py (both original and aCompCor).
  - Original:  {deriv}/fc_mni/{sub}_{roi}_{hemi}_loc_ppi_mni.nii.gz
  - aCompCor:  {deriv}/fc_mni/{sub}_{roi}_{hemi}_loc_ppi_acompcor_mni.nii.gz

Run: python dice_compare_acompcor.py
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import stats

study_dir = "/lab_data/behrmannlab/vlad/ptoc"
results_dir = '/user_data/csimmon2/git_repos/ptoc/results'

sub_info = pd.read_csv('/user_data/csimmon2/git_repos/ptoc/sub_info.csv')
subs = sub_info[sub_info['group'] == 'control']['sub'].tolist()
rois = ['pIPS', 'LO']
hemispheres = ['left', 'right']


def dice_coefficient(img1_data, img2_data):
    """Calculate Dice coefficient between two binary arrays."""
    img1_bin = (img1_data > 0).astype(int)
    img2_bin = (img2_data > 0).astype(int)
    intersection = np.sum(img1_bin * img2_bin)
    total = np.sum(img1_bin) + np.sum(img2_bin)
    if total == 0:
        return np.nan
    return 2.0 * intersection / total


def load_subject_maps(subs, suffix):
    """
    Load per-subject MNI maps, average across hemispheres per ROI.
    suffix: '_mni' for original, '_acompcor_mni' for aCompCor
    Returns: {sub: {roi: averaged_data}} and list of valid subjects
    """
    subject_data = {}
    valid_subjects = []

    for sub in subs:
        has_all = True
        subject_data[sub] = {}

        for roi in rois:
            hemi_arrays = []
            for hemi in hemispheres:
                f = f'{study_dir}/{sub}/ses-01/derivatives/fc_mni/{sub}_{roi}_{hemi}_loc_ppi{suffix}.nii.gz'
                if os.path.exists(f):
                    hemi_arrays.append(nib.load(f).get_fdata())
                else:
                    print(f"  Missing: {f}")
                    has_all = False

            if len(hemi_arrays) == 2:
                subject_data[sub][roi] = (hemi_arrays[0] + hemi_arrays[1]) / 2
            else:
                has_all = False

        if has_all:
            valid_subjects.append(sub)

    return subject_data, valid_subjects


def compute_dice_for_pipeline(subject_data, valid_subs):
    """
    Compute within-subject (pIPS vs LO) and between-subject Dice.
    Returns: dict with 'within' (per-subject D-V Dice), 
             'between_dorsal' (per-subject avg pairwise), 
             'between_ventral' (per-subject avg pairwise)
    """
    # Within-subject: pIPS vs LO
    within = {}
    for sub in valid_subs:
        within[sub] = dice_coefficient(subject_data[sub]['pIPS'], subject_data[sub]['LO'])

    # Between-subject: same ROI across subjects
    between = {roi: {} for roi in rois}
    for roi in rois:
        for sub in valid_subs:
            pairs = []
            for other in valid_subs:
                if sub != other:
                    pairs.append(dice_coefficient(subject_data[sub][roi], subject_data[other][roi]))
            between[roi][sub] = np.mean(pairs)

    return {
        'within': within,
        'between_dorsal': between['pIPS'],
        'between_ventral': between['LO']
    }


def summarize(dice_dict, label):
    """Print summary stats for one pipeline."""
    within_vals = list(dice_dict['within'].values())
    dorsal_vals = list(dice_dict['between_dorsal'].values())
    ventral_vals = list(dice_dict['between_ventral'].values())

    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    for name, vals in [('Within-sub (D-V)', within_vals),
                       ('Between-sub dorsal', dorsal_vals),
                       ('Between-sub ventral', ventral_vals)]:
        m = np.mean(vals)
        ci = stats.t.interval(0.95, len(vals)-1, loc=m, scale=stats.sem(vals)) if len(vals) > 1 else (m, m)
        print(f"  {name:25s}: {m:.4f}  95%CI [{ci[0]:.4f}, {ci[1]:.4f}]  (n={len(vals)})")


# ---- MAIN ----
print("Loading ORIGINAL PPI MNI maps...")
orig_data, orig_subs = load_subject_maps(subs, '_mni')
print(f"  Valid: {len(orig_subs)} subjects")

print("\nLoading ACOMPCOR PPI MNI maps...")
acomp_data, acomp_subs = load_subject_maps(subs, '_acompcor_mni')
print(f"  Valid: {len(acomp_subs)} subjects")

# Use intersection of valid subjects
common_subs = sorted(set(orig_subs) & set(acomp_subs))
print(f"\nCommon valid subjects: {len(common_subs)}: {common_subs}")

if len(common_subs) == 0:
    print("ERROR: No subjects have both original and aCompCor MNI maps.")
    print("Run native2mni.py for both pipelines first.")
    exit(1)

orig_dice = compute_dice_for_pipeline(orig_data, common_subs)
acomp_dice = compute_dice_for_pipeline(acomp_data, common_subs)

summarize(orig_dice, 'ORIGINAL (5 sub)')
summarize(acomp_dice, 'ACOMPCOR (5 sub)')

# ---- PAIRED COMPARISON TABLE ----
print(f"\n{'='*70}")
print("  PAIRED COMPARISON: within-subject Dice (pIPS vs LO)")
print(f"{'='*70}")
print(f"  {'Subject':10s} {'Original':>10s} {'aCompCor':>10s} {'Diff':>10s}")
print(f"  {'-'*40}")
diffs = []
for sub in common_subs:
    o = orig_dice['within'][sub]
    a = acomp_dice['within'][sub]
    d = a - o
    diffs.append(d)
    print(f"  {sub:10s} {o:10.4f} {a:10.4f} {d:10.4f}")

print(f"  {'-'*40}")
print(f"  {'Mean diff':10s} {'':>10s} {'':>10s} {np.mean(diffs):10.4f}")
print(f"  {'Max |diff|':10s} {'':>10s} {'':>10s} {np.max(np.abs(diffs)):10.4f}")

if len(diffs) > 1:
    t, p = stats.ttest_rel(
        [orig_dice['within'][s] for s in common_subs],
        [acomp_dice['within'][s] for s in common_subs]
    )
    print(f"\n  Paired t-test: t({len(common_subs)-1}) = {t:.4f}, p = {p:.4f}")

# ---- SAVE ----
out_dir = f'{results_dir}/acompcor_comparison'
os.makedirs(out_dir, exist_ok=True)

rows = []
for sub in common_subs:
    rows.append({
        'subject': sub,
        'dice_original': orig_dice['within'][sub],
        'dice_acompcor': acomp_dice['within'][sub],
        'diff': acomp_dice['within'][sub] - orig_dice['within'][sub],
        'between_dorsal_orig': orig_dice['between_dorsal'][sub],
        'between_dorsal_acomp': acomp_dice['between_dorsal'][sub],
        'between_ventral_orig': orig_dice['between_ventral'][sub],
        'between_ventral_acomp': acomp_dice['between_ventral'][sub],
    })
df = pd.DataFrame(rows)
csv_path = f'{out_dir}/dice_original_vs_acompcor_5sub.csv'
df.to_csv(csv_path, index=False)
print(f"\nSaved: {csv_path}")