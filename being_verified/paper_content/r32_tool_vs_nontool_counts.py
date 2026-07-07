#!/usr/bin/env python3
"""
R3.2 / Figure 7: count significant clusters & voxels, tools-vs-non-tools, per seed.

Matches Fig 7 exactly by reusing the recipe in threshold_tools_vs_nontools.py:
mean of the 18 per-subject MNI maps -> z-score across voxels -> two-sided FDR
(alpha=.05, cluster>5). Positive = tools>non-tools, negative = non-tools>tools.

If you already ran threshold_tools_vs_nontools.py, it reuses the saved
*_thresh.nii.gz; otherwise it rebuilds the group map from the subject maps.

Run in the `fmri` env:  python fig7_count_tools_vs_nontools.py
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import image
from nilearn.glm import threshold_stats_img
from scipy import ndimage

curr_dir      = "/user_data/csimmon2/git_repos/ptoc"
study_dir     = "/lab_data/behrmannlab/vlad/ptoc"
group_out_dir = f"{curr_dir}/results/tools/group_tools_vs_nontools"
sub_info_path = f"{curr_dir}/sub_info_tool.csv"

ALPHA        = 0.05
CLUSTER_MIN  = 5      # voxels, same as the figure
rois         = ["pIPS", "LO"]
hemis        = ["left", "right"]

subs = pd.read_csv(sub_info_path)
subs = subs[subs["exp"] == "spaceloc"]["sub"].tolist()


def get_thresh_map(roi, hemi):
    """Return the thresholded z-map (pos=tools>nontools, neg=nontools>tools)."""
    saved = f"{group_out_dir}/{roi}_{hemi}_tools_vs_nontools_ppi_thresh.nii.gz"
    if os.path.exists(saved):
        return nib.load(saved)

    # rebuild with the same recipe as threshold_tools_vs_nontools.py
    imgs = []
    for sub in subs:
        f = (f"{study_dir}/{sub}/ses-01/derivatives/ppi/mni/"
             f"{sub}_{roi}_{hemi}_tools_vs_nontools_ToolLoc_ppi_mni.nii.gz")
        if os.path.exists(f):
            im = image.load_img(f)
            if im.shape == (91, 109, 91):
                imgs.append(im)
    if not imgs:
        print(f"  [!] no subject maps found for {roi} {hemi}")
        return None
    avg   = image.mean_img(imgs)
    zstat = image.math_img("(img-img.mean())/img.std()", img=avg)
    _, thr = threshold_stats_img(zstat, alpha=ALPHA, height_control="fdr",
                                 cluster_threshold=CLUSTER_MIN, two_sided=True)
    data = zstat.get_fdata()
    data[np.abs(data) < thr] = 0
    return nib.Nifti1Image(data.astype("double"), zstat.affine)


def count_dir(mask):
    """Clusters (>=CLUSTER_MIN voxels) and total voxels in a boolean mask."""
    lab, n = ndimage.label(mask)
    if n == 0:
        return 0, 0
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    keep  = sizes[sizes >= CLUSTER_MIN]
    return int(len(keep)), int(keep.sum())


print(f"{len(subs)} subjects | two-sided FDR alpha={ALPHA}, cluster>={CLUSTER_MIN}\n")
for roi in rois:
    pos_any = None  # tools > non-tools, union across hemispheres
    neg_any = None  # non-tools > tools
    for hemi in hemis:
        m = get_thresh_map(roi, hemi)
        if m is None:
            continue
        d = m.get_fdata()
        p, n = (d > 0), (d < 0)
        pos_any = p if pos_any is None else (pos_any | p)
        neg_any = n if neg_any is None else (neg_any | n)

    pc, pv = count_dir(pos_any) if pos_any is not None else (0, 0)
    nc, nv = count_dir(neg_any) if neg_any is not None else (0, 0)
    print(f"[{roi} seed]  (both hemispheres combined)")
    print(f"  tools > non-tools : {pc} clusters, {pv} voxels")
    print(f"  non-tools > tools : {nc} clusters, {nv} voxels\n")
