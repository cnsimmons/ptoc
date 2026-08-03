"""
Supp Fig 3 (R2.3): pFS generalizability — Fig 3D format.
Three bars: between-subject pIPS, between-subject pFS, within-subject pIPS-pFS.
Save to acompcor_comparison/.
"""
import os
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from scipy.stats import ttest_rel

study_dir = "/lab_data/behrmannlab/vlad/ptoc"
results_dir = "/user_data/csimmon2/git_repos/ptoc/results/acompcor_comparison"
sub_info = pd.read_csv("/user_data/csimmon2/git_repos/ptoc/sub_info.csv")
subs = [s for s in sub_info[sub_info["group"] == "control"]["sub"].tolist() if s != "sub-084"]
hemispheres = ["left", "right"]

SUFFIX = {
    "pIPS": "_loc_ppi_mni.nii.gz",
    "PFS":  "_loc_ppi_acompcor_mni.nii.gz",
}

def dice(a, b):
    a, b = (a > 0).astype(int), (b > 0).astype(int)
    tot = a.sum() + b.sum()
    return np.nan if tot == 0 else 2.0 * (a * b).sum() / tot

def load_maps(subs):
    data, valid = {}, []
    for sub in subs:
        data[sub], ok = {}, True
        for roi, suf in SUFFIX.items():
            arrs = []
            for hemi in hemispheres:
                f = f"{study_dir}/{sub}/ses-01/derivatives/fc_mni/{sub}_{roi}_{hemi}{suf}"
                if os.path.exists(f):
                    arrs.append(nib.load(f).get_fdata())
                else:
                    print(f"  missing: {f}"); ok = False
            if len(arrs) == 2:
                data[sub][roi] = (arrs[0] + arrs[1]) / 2
            else:
                ok = False
        if ok:
            valid.append(sub)
    return data, valid

print("Loading maps...")
data, valid = load_maps(subs)
n = len(valid)
print(f"Valid subjects: {n}")

# Within-subject pIPS-pFS
within = np.array([dice(data[s]["pIPS"], data[s]["PFS"]) for s in valid])

# Between-subject
def between(roi):
    out = []
    for s in valid:
        out.append(np.mean([dice(data[s][roi], data[o][roi]) for o in valid if o != s]))
    return np.array(out)

bt_dorsal = between("pIPS")
bt_pfs = between("PFS")

# Stats
t_arc = lambda x: np.arcsin(np.sqrt(x))
t1, p1 = ttest_rel(t_arc(within), t_arc(bt_dorsal))
t2, p2 = ttest_rel(t_arc(within), t_arc(bt_pfs))
print(f"\nWithin pIPS-pFS mean: {within.mean():.3f}")
print(f"Between pIPS mean:   {bt_dorsal.mean():.3f}")
print(f"Between pFS mean:    {bt_pfs.mean():.3f}")
print(f"Within vs bt-pIPS: t({n-1})={t1:.2f}, p={p1:.4f}")
print(f"Within vs bt-pFS:  t({n-1})={t2:.2f}, p={p2:.4f}")

# Plot — matching dice_figure_stats.py style
labels = ["between-subj\npIPS", "between-subj\npFS", "within-subj\npIPS\u2013pFS"]
means = [bt_dorsal.mean(), bt_pfs.mean(), within.mean()]
cis = [1.96 * x.std() / np.sqrt(n) for x in [bt_dorsal, bt_pfs, within]]
dots = [bt_dorsal, bt_pfs, within]
colors = ["#4ac0c0", "#ff9b83", "#b39ddb"]

fig, ax = plt.subplots(figsize=(5, 4.5))
for i, (m, ci, d, c) in enumerate(zip(means, cis, dots, colors)):
    face = to_rgba(c, alpha=0.3)
    ax.bar(i, m, yerr=ci, capsize=4, color=face, edgecolor=c, linewidth=1.5)
    ax.scatter(np.full(n, i) + np.random.uniform(-0.15, 0.15, n), d, color="gray", s=20, alpha=0.6, zorder=3)
ax.set_xticks(range(3))
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Dice coefficient", fontsize=11)
ax.set_ylim(0, 1)
ax.set_title("PPI network overlap: pFS", fontsize=11)
plt.tight_layout()
fig.savefig(f"{results_dir}/supp_fig_pFS_fig3d.png", dpi=300, bbox_inches="tight")
print(f"\nSaved: {results_dir}/supp_fig_pFS_fig3d.png")