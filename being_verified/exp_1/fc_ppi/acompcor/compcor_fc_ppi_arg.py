"""
aCompCor PPI network-overlap: figure + stats (R2.2 V1 control, R2.3 pFS).

Stats: arcsine-sqrt Dice -> one-way repeated-measures ANOVA (single factor = pair,
       5 levels). Object-vs-control tested two ways on top of that ANOVA:
       (1) planned contrast, object pairs (+2 each) vs control pairs (-3 each);
       (2) pairwise post-hocs (each object pair vs each V1 pair), Holm-corrected.
       Balanced: every subject has all 5 pairs. No averaging.

N=18 (sub-084 excluded). Run: python dice_figure_stats.py
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import stats
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests
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


# ---- load ----
data, valid = load_maps(subs)
print(f"\nValid subjects (all 4 ROIs, both hemis): {len(valid)}\n")

bt_dorsal  = between_roi(data, valid, "pIPS")
bt_ventral = between_roi(data, valid, "LO")
w_pips_lo  = within_pair(data, valid, "pIPS", "LO")
w_pfs_pips = within_pair(data, valid, "PFS", "pIPS")
w_pfs_lo   = within_pair(data, valid, "PFS", "LO")
w_v1_pips  = within_pair(data, valid, "V1", "pIPS")
w_v1_lo    = within_pair(data, valid, "V1", "LO")

# ---- figure ----
fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.5),
                               gridspec_kw={"width_ratios": [3, 4]})

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

# ---- stats: one-way RM-ANOVA (factor = pair), then object-vs-control ----
asin = lambda v: np.arcsin(np.sqrt(v))

pairs = {
    "pIPS-LO":  ("object",  w_pips_lo),
    "pFS-pIPS": ("object",  w_pfs_pips),
    "pFS-LO":   ("object",  w_pfs_lo),
    "V1-pIPS":  ("control", w_v1_pips),
    "V1-LO":    ("control", w_v1_lo),
}

rows = []
for name, (grp, vals) in pairs.items():
    for sub, v in zip(valid, vals):
        rows.append({"subject": sub, "pair": name, "dice_asin": asin(v)})
sdf = pd.DataFrame(rows)

aov = AnovaRM(sdf, depvar="dice_asin", subject="subject", within=["pair"]).fit()
print("\nOne-way RM-ANOVA (arcsine-sqrt Dice), factor = pair (5 levels):")
print(aov.anova_table)

# planned contrast: object pairs (+2) vs control pairs (-3), per subject
weights = {"pIPS-LO": 2, "pFS-pIPS": 2, "pFS-LO": 2, "V1-pIPS": -3, "V1-LO": -3}
contrast = np.zeros(len(valid))
for name, (grp, vals) in pairs.items():
    contrast += weights[name] * asin(vals)
t_c, p_c = stats.ttest_1samp(contrast, 0.0)
dz_c = contrast.mean() / contrast.std(ddof=1)
print(f"\nPlanned contrast object(+2) vs control(-3), n={len(valid)}: "
      f"t({len(valid)-1}) = {t_c:.2f}, p = {p_c:.2e}, dz = {dz_c:.2f}")

# pairwise post-hocs (each object pair vs each V1 pair), Holm-corrected
obj_pairs  = [k for k, (g, _) in pairs.items() if g == "object"]
ctrl_pairs = [k for k, (g, _) in pairs.items() if g == "control"]
comps, tvals, pvals, dzs = [], [], [], []
for op in obj_pairs:
    for cp in ctrl_pairs:
        o, c = asin(pairs[op][1]), asin(pairs[cp][1])
        d = o - c
        t, p = stats.ttest_rel(o, c)
        comps.append(f"{op} vs {cp}")
        tvals.append(t); pvals.append(p); dzs.append(d.mean() / d.std(ddof=1))
_, p_holm, _, _ = multipletests(pvals, method="holm")

print("\nPost-hoc (paired, Holm-corrected):")
print(f"  {'comparison':22s}{'t':>8s}{'p_holm':>11s}{'dz':>7s}")
for comp, t, ph, dz in zip(comps, tvals, p_holm, dzs):
    print(f"  {comp:22s}{t:8.2f}{ph:11.2e}{dz:7.2f}")

print("\nPair means (Dice):")
for name, (grp, vals) in pairs.items():
    print(f"  {name:9s} [{grp:7s}]: {np.nanmean(vals):.3f}")

# ---- save ----
aov.anova_table.to_csv(f"{out_dir}/dice_anova.csv")
pd.DataFrame({"comparison": comps, "t": tvals, "p_uncorrected": pvals,
              "p_holm": p_holm, "dz": dzs}).to_csv(
              f"{out_dir}/dice_posthoc.csv", index=False)
pd.DataFrame([{"contrast": "object(+2) vs control(-3)", "t": t_c,
               "df": len(valid) - 1, "p": p_c, "dz": dz_c}]).to_csv(
              f"{out_dir}/dice_contrast.csv", index=False)
pd.DataFrame({"subject": valid, "pIPS_LO": w_pips_lo, "PFS_pIPS": w_pfs_pips,
              "PFS_LO": w_pfs_lo, "V1_pIPS": w_v1_pips, "V1_LO": w_v1_lo,
              "between_dorsal": bt_dorsal, "between_ventral": bt_ventral}).to_csv(
              f"{out_dir}/dice_per_subject.csv", index=False)
print(f"\nSaved: dice_anova.csv, dice_posthoc.csv, dice_contrast.csv, "
      f"dice_per_subject.csv  (in {out_dir})")
