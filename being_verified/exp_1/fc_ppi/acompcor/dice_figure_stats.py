"""
aCompCor PPI network-overlap: figure + stats (R2.2 V1 control, R2.3 pFS).

RIGHT panel (new): within-subject Dice for 4 region pairs.
  Test: one-way RM-ANOVA (factor = pair, 4 levels) + planned group contrast
        object pairs (pFS-pIPS, pFS-LO) +1 vs control pairs (V1-pIPS, V1-LO) -1.
  Plus within-group post-hocs (pFS-pIPS vs pFS-LO; V1-pIPS vs V1-LO), Holm.
LEFT panel: Fig 3D reproduced on aCompCor maps (between-dorsal, between-ventral,
  within-subject dorsal-ventral). Test: RM-ANOVA (3 levels) + Holm post-hocs,
  within-DV vs each between-subject bar. Mirrors the manuscript.

Both arcsine-sqrt transformed. Balanced, no averaging. N=18 (sub-084 excluded).
Run: python dice_figure_stats.py
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
from matplotlib.colors import to_rgba
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
FILL_ALPHA = 0.35


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


def stars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."


def bar_with_dots(ax, x, vals, color, width=0.6):
    m, lo, hi = ci95(vals)
    ax.bar(x, m, width=width, facecolor=to_rgba(color, FILL_ALPHA),
           edgecolor=color, linewidth=2, zorder=1)
    ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="none",
                ecolor="#333333", elinewidth=1.5, capsize=4, zorder=3)
    jit = (np.random.RandomState(0).rand(len(vals)) - 0.5) * width * 0.5
    ax.scatter(np.full(len(vals), x) + jit, vals, s=14, color="#333333",
               alpha=0.45, zorder=2, linewidths=0)


def sig_bar(ax, x1, x2, y, text, h=0.015, lw=1.3, c="#333333"):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=lw, c=c, clip_on=False)
    ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom",
            fontsize=13, color=c, clip_on=False)


# ---- load ----
data, valid = load_maps(subs)
n = len(valid)
print(f"\nValid subjects (all 4 ROIs, both hemis): {n}\n")

bt_dorsal  = between_roi(data, valid, "pIPS")
bt_ventral = between_roi(data, valid, "LO")
w_pips_lo  = within_pair(data, valid, "pIPS", "LO")
w_pfs_pips = within_pair(data, valid, "PFS", "pIPS")
w_pfs_lo   = within_pair(data, valid, "PFS", "LO")
w_v1_pips  = within_pair(data, valid, "V1", "pIPS")
w_v1_lo    = within_pair(data, valid, "V1", "LO")

asin = lambda v: np.arcsin(np.sqrt(v))

# ---- LEFT-panel stats: Fig 3D reproduction (3 levels) ----
left = {"between_dorsal": bt_dorsal, "between_ventral": bt_ventral, "within_DV": w_pips_lo}
rowsL = [{"subject": s, "cond": k, "y": asin(v)}
         for k, arr in left.items() for s, v in zip(valid, arr)]
aovL = AnovaRM(pd.DataFrame(rowsL), "y", "subject", within=["cond"]).fit()

pL_names, tL, pL = [], [], []
for k in ["between_dorsal", "between_ventral"]:
    t, p = stats.ttest_rel(asin(w_pips_lo), asin(left[k]))
    pL_names.append(f"within_DV vs {k}"); tL.append(t); pL.append(p)
_, pL_holm, _, _ = multipletests(pL, method="holm")

print("LEFT panel (Fig 3D, aCompCor) — RM-ANOVA (3 levels):")
print(aovL.anova_table)
print("  post-hoc (Holm):")
for name, t, ph in zip(pL_names, tL, pL_holm):
    print(f"    {name:32s} t={t:6.2f}  p_holm={ph:.2e}")

# ---- RIGHT-panel stats: object vs control (4 levels) ----
right = {"pFS-pIPS": w_pfs_pips, "pFS-LO": w_pfs_lo,
         "V1-pIPS": w_v1_pips, "V1-LO": w_v1_lo}
rowsR = [{"subject": s, "pair": k, "y": asin(v)}
         for k, arr in right.items() for s, v in zip(valid, arr)]
aovR = AnovaRM(pd.DataFrame(rowsR), "y", "subject", within=["pair"]).fit()

wts = {"pFS-pIPS": 1, "pFS-LO": 1, "V1-pIPS": -1, "V1-LO": -1}
contrast = sum(wts[k] * asin(arr) for k, arr in right.items())
t_c, p_c = stats.ttest_1samp(contrast, 0.0)
dz_c = contrast.mean() / contrast.std(ddof=1)

print("\nRIGHT panel — RM-ANOVA (4 levels), factor = pair:")
print(aovR.anova_table)
print(f"\nGroup contrast object(+1) vs control(-1), n={n}: "
      f"t({n-1}) = {t_c:.2f}, p = {p_c:.2e}, dz = {dz_c:.2f}")

# within-group post-hocs (do the two object pairs / two control pairs differ?)
within_tests = [("pFS-pIPS", "pFS-LO"), ("V1-pIPS", "V1-LO")]
wc, wt, wp, wdz = [], [], [], []
for a, b in within_tests:
    oa, ob = asin(right[a]), asin(right[b])
    d = oa - ob
    t, p = stats.ttest_rel(oa, ob)
    wc.append(f"{a} vs {b}"); wt.append(t); wp.append(p)
    wdz.append(d.mean() / d.std(ddof=1))
_, wp_holm, _, _ = multipletests(wp, method="holm")

print("\nWithin-group post-hocs (paired, Holm-corrected):")
print(f"  {'comparison':22s}{'t':>8s}{'p_holm':>11s}{'dz':>7s}")
for c, t, ph, dz in zip(wc, wt, wp_holm, wdz):
    print(f"  {c:22s}{t:8.2f}{ph:11.2e}{dz:7.2f}")

print("\nPair means (Dice):")
for k, arr in {**left, **right}.items():
    print(f"  {k:16s}: {np.nanmean(arr):.3f}")

# ---- figure ----
fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.5),
                               gridspec_kw={"width_ratios": [3, 4]})

Ldef = [("between-subj\ndorsal", bt_dorsal, TEAL),
        ("between-subj\nventral", bt_ventral, PINK),
        ("within-subj\ndorsal-ventral", w_pips_lo, PURPLE)]
for i, (lab, v, c) in enumerate(Ldef):
    bar_with_dots(axL, i, v, c)
axL.set_xticks(range(len(Ldef)))
axL.set_xticklabels([l for l, _, _ in Ldef], fontsize=9)
axL.set_ylabel("Dice coefficient")
axL.set_title("Network overlap (Fig 3D, aCompCor)", fontsize=10)
sig_bar(axL, 1, 2, 1.00, stars(pL_holm[1]))   # within vs between-ventral (shorter)
sig_bar(axL, 0, 2, 1.07, stars(pL_holm[0]))   # within vs between-dorsal (longer)

Rdef = [("pFS-pIPS", w_pfs_pips, PURPLE),
        ("pFS-LO", w_pfs_lo, PURPLE),
        ("V1-pIPS", w_v1_pips, GRAY),
        ("V1-LO", w_v1_lo, GRAY)]
for i, (lab, v, c) in enumerate(Rdef):
    bar_with_dots(axR, i, v, c)
axR.set_xticks(range(len(Rdef)))
axR.set_xticklabels([l for l, _, _ in Rdef], fontsize=9)
axR.set_title("Within-subject region-pair overlap", fontsize=10)
axR.legend(handles=[Patch(facecolor=to_rgba(PURPLE, FILL_ALPHA), edgecolor=PURPLE,
                          linewidth=2, label="object-object"),
                    Patch(facecolor=to_rgba(GRAY, FILL_ALPHA), edgecolor=GRAY,
                          linewidth=2, label="control (V1)")],
           frameon=False, fontsize=9, loc="upper right")
sig_bar(axR, 0.5, 2.5, 1.00, stars(p_c))       # object group vs control group

for ax in (axL, axR):
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(axis="y", alpha=0.2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

plt.tight_layout()
fig_path = f"{out_dir}/dice_overlap_figure.png"
fig.savefig(fig_path, dpi=300, bbox_inches="tight")
print(f"\nSaved figure: {fig_path}")

# ---- save stats ----
aovR.anova_table.to_csv(f"{out_dir}/dice_anova_right.csv")
aovL.anova_table.to_csv(f"{out_dir}/dice_anova_left.csv")
pd.DataFrame([{"contrast": "object(+1) vs control(-1)", "t": t_c,
               "df": n - 1, "p": p_c, "dz": dz_c}]).to_csv(
              f"{out_dir}/dice_contrast.csv", index=False)
pd.DataFrame({"comparison": wc, "t": wt, "p_uncorrected": wp,
              "p_holm": wp_holm, "dz": wdz}).to_csv(
              f"{out_dir}/dice_within_group_posthoc.csv", index=False)
pd.DataFrame({"subject": valid, "pIPS_LO": w_pips_lo, "PFS_pIPS": w_pfs_pips,
              "PFS_LO": w_pfs_lo, "V1_pIPS": w_v1_pips, "V1_LO": w_v1_lo,
              "between_dorsal": bt_dorsal, "between_ventral": bt_ventral}).to_csv(
              f"{out_dir}/dice_per_subject.csv", index=False)
print("Saved: dice_anova_right.csv, dice_anova_left.csv, dice_contrast.csv, "
      "dice_within_group_posthoc.csv, dice_per_subject.csv")
