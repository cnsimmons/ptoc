"""
aCompCor PPI network-overlap: figure + stats (R2.2 V1 control, R2.3 pFS).

Left panel  — Fig 3D recomputed on aCompCor maps:
              between-subject dorsal (pIPS), between-subject ventral (LO),
              within-subject dorsal-ventral (pIPS-LO).
Right panel — within-subject region-pair Dice:
              object pairs (pIPS-LO, pFS-pIPS, pFS-LO) vs control (V1-pIPS, V1-LO).

Prints/saves 4 paired arcsine-sqrt t-tests. N=18 (sub-084 excluded).
Run: python dice_figure_stats.py
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

study_dir = "/lab_data/behrmannlab/vlad/ptoc"
results_dir = "/user_data/csimmon2/git_repos/ptoc/results"
out_dir = f"{results_dir}/acompcor_comparison"
os.makedirs(out_dir, exist_ok=True)

sub_info = pd.read_csv("/user_data/csimmon2/git_repos/ptoc/sub_info.csv")
subs = sub_info[sub_info["group"] == "control"]["sub"].tolist()
subs = [s for s in subs if s != "sub-084"]        # documented exclusion (N=18)

rois = ["pIPS", "LO", "PFS", "V1"]
hemispheres = ["left", "right"]
TEAL, PINK, PURPLE, GRAY = "#4ac0c0", "#ff9b83", "#9467bd", "#9aa0a6"


def dice(a, b):
    a, b = (a > 0).astype(int), (b > 0).astype(int)
    tot = a.sum() + b.sum()
    return np.nan if tot == 0 else 2.0 * (a * b).sum() / tot


def load_maps(subs):
    data, valid = {}, []
    for sub in subs:
        data[sub], ok = {}, True
        for roi in rois:
            arrs = []
            for hemi in hemispheres:
                f = f"{study_dir}/{sub}/ses-01/derivatives/fc_mni/{sub}_{roi}_{hemi}_loc_ppi_acompcor_mni.nii.gz"
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


def within_pair(data, valid, a, b):
    return np.array([dice(data[s][a], data[s][b]) for s in valid])


def between_roi(data, valid, roi):
    vals = []
    for s in valid:
        pw = [dice(data[s][roi], data[o][roi]) for o in valid if o != s]
        vals.append(np.mean(pw))
    return np.array(vals)


def ci95(v):
    v = v[~np.isnan(v)]
    m = v.mean()
    lo, hi = stats.t.interval(0.95, len(v) - 1, loc=m, scale=stats.sem(v))
    return m, lo, hi


def bar_with_dots(ax, x, vals, color, width=0.6):
    m, lo, hi = ci95(vals)
    ax.bar(x, m, width=width, color=color, edgecolor="none", zorder=1)
    ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="none",
                ecolor="#333333", elinewidth=1.5, capsize=4, zorder=3)
    jit = (np.random.RandomState(0).rand(len(vals)) - 0.5) * width * 0.5
    ax.scatter(np.full(len(vals), x) + jit, vals, s=14, color="#333333",
               alpha=0.45, zorder=2, linewidths=0)


data, valid = load_maps(subs)
print(f"\nValid subjects (all 4 ROIs, both hemis): {len(valid)}\n")

bt_dorsal  = between_roi(data, valid, "pIPS")
bt_ventral = between_roi(data, valid, "LO")
w_pips_lo  = within_pair(data, valid, "pIPS", "LO")
w_pfs_pips = within_pair(data, valid, "PFS", "pIPS")
w_pfs_lo   = within_pair(data, valid, "PFS", "LO")
w_v1_pips  = within_pair(data, valid, "V1", "pIPS")
w_v1_lo    = within_pair(data, valid, "V1", "LO")

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.5),
                               gridspec_kw={"width_ratios": [3, 5]})

L = [("between-subj\ndorsal", bt_dorsal, TEAL),
     ("between-subj\nventral", bt_ventral, PINK),
     ("within-subj\ndorsal-ventral", w_pips_lo, PURPLE)]
for i, (lab, v, c) in enumerate(L):
    bar_with_dots(axL, i, v, c)
axL.set_xticks(range(len(L)))
axL.set_xticklabels([l for l, _, _ in L], fontsize=9)
axL.set_ylabel("Dice coefficient")
axL.set_title("Network overlap (Fig 3D, aCompCor)", fontsize=10)

R = [("pFS-pIPS", w_pfs_pips, PURPLE),
     ("pFS-LO", w_pfs_lo, PURPLE),
     ("V1-pIPS", w_v1_pips, GRAY),
     ("V1-LO", w_v1_lo, GRAY)]

for i, (lab, v, c) in enumerate(R):
    bar_with_dots(axR, i, v, c)
axR.set_xticks(range(len(R)))
axR.set_xticklabels([l for l, _, _ in R], fontsize=9)
axR.set_title("Within-subject region-pair overlap", fontsize=10)
axR.legend(handles=[Patch(color=PURPLE, label="object-object"),
                    Patch(color=GRAY, label="control (V1)")],
           frameon=False, fontsize=9, loc="upper right")

for ax in (axL, axR):
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

plt.tight_layout()
fig_path = f"{out_dir}/dice_overlap_figure.png"
fig.savefig(fig_path, dpi=300, bbox_inches="tight")
print(f"Saved figure: {fig_path}")

asin = lambda v: np.arcsin(np.sqrt(v))


def paired(obj_v, ctrl_v, obj_name, ctrl_name):
    o, c = asin(obj_v), asin(ctrl_v)
    d = o - c
    t, p = stats.ttest_rel(o, c)
    return {"comparison": f"{obj_name} vs {ctrl_name}",
            "obj_mean": obj_v.mean(), "ctrl_mean": ctrl_v.mean(),
            "t": t, "df": len(obj_v) - 1, "p": p,
            "dz": d.mean() / d.std(ddof=1)}


tests = [
    paired(w_pips_lo,  w_v1_pips, "pIPS-LO", "V1-pIPS"),
    paired(w_pips_lo,  w_v1_lo,   "pIPS-LO", "V1-LO"),
    paired(w_pfs_pips, w_v1_pips, "pFS-pIPS", "V1-pIPS"),
    paired(w_pfs_lo,   w_v1_lo,   "pFS-LO", "V1-LO"),
]

print(f"\n{'comparison':22s}{'obj':>7s}{'ctrl':>7s}{'t':>8s}{'p':>11s}{'dz':>7s}")
for r in tests:
    print(f"{r['comparison']:22s}{r['obj_mean']:7.3f}{r['ctrl_mean']:7.3f}"
          f"{r['t']:8.2f}{r['p']:11.2e}{r['dz']:7.2f}")

pd.DataFrame(tests).to_csv(f"{out_dir}/dice_stats.csv", index=False)
pd.DataFrame({"subject": valid, "between_dorsal": bt_dorsal,
              "between_ventral": bt_ventral, "pIPS_LO": w_pips_lo,
              "PFS_pIPS": w_pfs_pips, "PFS_LO": w_pfs_lo,
              "V1_pIPS": w_v1_pips, "V1_LO": w_v1_lo}).to_csv(
              f"{out_dir}/dice_per_subject.csv", index=False)
print(f"\nSaved: dice_stats.csv, dice_per_subject.csv  (in {out_dir})")
