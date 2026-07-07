#!/usr/bin/env python3
"""
R3.2 / Figure 7 quantification: tools-vs-non-tools parcel counts, per seed.

Adapted from sim_diff_ex2 cell 7. Everything is the same as the Fig 6 analysis
(merged Schaefer-Wang-Julian atlas, bilateral seed averaging, 10k bootstrap CI +
LOO sign-consistency), EXCEPT the contrast:
    Fig 6 counts  (pIPS - LO)      within a condition
    here we count (tools - nontools) within each seed

It reuses the per-condition PPI maps you already have. Because each per-condition
PPI is contrasted against scramble (tool>scramble, nontool>scramble), differencing
them cancels scramble and yields the tool-vs-non-tool contrast shown in Figure 7.

Run in the `fmri` env:  python r32_tool_vs_nontool_counts.py
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.maskers import NiftiLabelsMasker
from sklearn.utils import resample
from scipy import stats

# ----- paths (identical to cell 7) -----
study_dir     = "/lab_data/behrmannlab/vlad/ptoc"
results_dir   = "/user_data/csimmon2/git_repos/ptoc/results"
sub_info_path = "/user_data/csimmon2/git_repos/ptoc/sub_info_tool.csv"

ANALYSIS   = "ppi"     # Fig 7 is task-based (PPI). Set "fc" for the correlation version.
N_BOOT     = 10000
LOO_THRESH = 0.75

# ----- atlas + masker -----
atlas_img    = nib.load(f"{results_dir}/exp2/schaefer_wang_merged_exp2.nii.gz")
atlas_labels = np.load(f"{results_dir}/exp2/merged_atlas_labels_exp2.npy", allow_pickle=True)
masker       = NiftiLabelsMasker(labels_img=atlas_img, standardize=False)

subs  = pd.read_csv(sub_info_path)
subs  = subs[subs["exp"] == "spaceloc"]["sub"].tolist()
hemis = ["left", "right"]

# exclude the functional Wang seeds (self-connections), exactly as cell 7 does
mask = np.ones(len(atlas_labels), dtype=bool)
for i, l in enumerate(atlas_labels):
    if "Wang_pIPS" in str(l) or "Wang_LO" in str(l):
        mask[i] = False


def cond_path(sub, roi, hemi, cond):
    # cell 7 naming: tools has NO condition token in the filename; nontools does
    if cond == "tools":
        return (f"{study_dir}/{sub}/ses-01/derivatives/{ANALYSIS}/mni/"
                f"{sub}_{roi}_{hemi}_ToolLoc_{ANALYSIS}_mni.nii.gz")
    return (f"{study_dir}/{sub}/ses-01/derivatives/{ANALYSIS}/mni/"
            f"{sub}_{roi}_{hemi}_{cond}_ToolLoc_{ANALYSIS}_mni.nii.gz")


def load_seed(sub, roi, cond):
    """Bilateral-averaged parcel vector for one subject/seed/condition."""
    combined, n = None, 0
    for hemi in hemis:
        f = cond_path(sub, roi, hemi, cond)
        if os.path.exists(f):
            v = masker.fit_transform(nib.load(f))[0]
            combined = v if combined is None else combined + v
            n += 1
        else:
            print(f"  missing: {f}")
    return combined / n if n > 0 else None


def analyze(seed):
    T, N = [], []
    for sub in subs:
        t  = load_seed(sub, seed, "tools")
        nt = load_seed(sub, seed, "nontools")
        if t is not None and nt is not None:
            T.append(t[mask]); N.append(nt[mask])
    T, N = np.array(T), np.array(N)
    k = len(T)
    if k < 3:
        print(f"[{seed}] only {k} subjects with both conditions — skipping")
        return

    diff      = T - N                     # subjects x parcels (tools - nontools)
    mean_diff = diff.mean(0)

    # bootstrap 95% CI across subjects
    boot = np.array([diff[resample(range(k), replace=True, n_samples=k)].mean(0)
                     for _ in range(N_BOOT)])
    lo, hi   = np.percentile(boot, 2.5, 0), np.percentile(boot, 97.5, 0)
    sig_boot = (lo > 0) | (hi < 0)

    # LOO sign-consistency (>= 0.75), as in cell 7
    loo = np.zeros(diff.shape[1])
    for j in range(k):
        tr = np.delete(diff, j, 0).mean(0)
        loo += (np.sign(tr) == np.sign(diff[j])).astype(float)
    sig_loo = (loo / k) >= LOO_THRESH

    sig          = sig_boot & sig_loo
    tool_bias    = int(np.sum(sig & (mean_diff > 0)))
    nontool_bias = int(np.sum(sig & (mean_diff < 0)))

    obs = np.array([tool_bias, nontool_bias])
    if obs.sum() > 0:
        chi2, p = stats.chisquare(obs, [obs.sum() / 2] * 2)
    else:
        chi2, p = float("nan"), float("nan")

    print(f"\n[{seed} seed]  (n = {k} subjects)")
    print(f"  tools > non-tools : {tool_bias} parcels")
    print(f"  non-tools > tools : {nontool_bias} parcels")
    print(f"  chi2 = {chi2:.3f}, p = {p:.4g}")


if __name__ == "__main__":
    print(f"ANALYSIS = {ANALYSIS} | n_boot = {N_BOOT} | loo >= {LOO_THRESH}")
    print(f"{len(subs)} subjects | {int(mask.sum())} parcels after excluding Wang seeds")
    for seed in ["pIPS", "LO"]:
        analyze(seed)
