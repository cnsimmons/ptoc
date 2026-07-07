"""
aCompCor PPI network-overlap: three separate figures + stats.

Figure 1 (aCompCor robustness): Fig 3D reproduced on aCompCor maps
  3 bars: between-dorsal, between-ventral, within dorsal-ventral
  RM-ANOVA + Holm post-hocs.

Supp Figure pFS: within-subject Dice for pIPS-LO (reference), pFS-pIPS, pFS-LO

Supp Figure V1: within-subject Dice for pIPS-LO (reference), V1-pIPS, V1-LO

Arcsine-sqrt transformed for stats. N=18 (sub-084 excluded).
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


# ---- helpers ----

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


def style_ax(ax, ylim_top=1.1):
    ax.set_ylim(0, ylim_top)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("Dice coefficient")
    ax.grid(axis="y", alpha=0.2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


asin = lambda v: np.arcsin(np.sqrt(v))


# ---- load ----
data, valid = load_maps(subs)
n = len(valid)
print(f"\nValid subjects (all 4 ROIs, both hemis): {n}\n")

# ---- compute all arrays ----
bt_dorsal  = between_roi(data, valid, "pIPS")
bt_ventral = between_roi(data, valid, "LO")
w_pips_lo  = within_pair(data, valid, "pIPS", "LO")
w_pfs_pips = within_pair(data, valid, "PFS", "pIPS")
w_pfs_lo   = within_pair(data, valid, "PFS", "LO")
w_v1_pips  = within_pair(data, valid, "V1", "pIPS")
w_v1_lo    = within_pair(data, valid, "V1", "LO")


# ================================================================
# STATS: Fig 1 (aCompCor robustness) — RM-ANOVA, 3 levels
# ================================================================
left = {"between_dorsal": bt_dorsal, "between_ventral": bt_ventral, "within_DV": w_pips_lo}
rowsL = [{"subject": s, "cond": k, "y": asin(v)}
         for k, arr in left.items() for s, v in zip(valid, arr)]
aovL = AnovaRM(pd.DataFrame(rowsL), "y", "subject", within=["cond"]).fit()

pL_names, tL, pL = [], [], []
for k in ["between_dorsal", "between_ventral"]:
    t, p = stats.ttest_rel(asin(w_pips_lo), asin(left[k]))
    pL_names.append(f"within_DV vs {k}"); tL.append(t); pL.append(p)
_, pL_holm, _, _ = multipletests(pL, method="holm")

print("Fig 1 (aCompCor robustness) — RM-ANOVA (3 levels):")
print(aovL.anova_table)
print("  post-hoc (Holm):")
for name, t, ph in zip(pL_names, tL, pL_holm):
    print(f"    {name:32s} t={t:6.2f}  p_holm={ph:.2e}")


# ================================================================
# STATS: 4-level ANOVA + planned contrast (kept for CSV export)
# ================================================================
right = {"pFS-pIPS": w_pfs_pips, "pFS-LO": w_pfs_lo,
         "V1-pIPS": w_v1_pips, "V1-LO": w_v1_lo}
rowsR = [{"subject": s, "pair": k, "y": asin(v)}
         for k, arr in right.items() for s, v in zip(valid, arr)]
aovR = AnovaRM(pd.DataFrame(rowsR), "y", "subject", within=["pair"]).fit()

wts = {"pFS-pIPS": 1, "pFS-LO": 1, "V1-pIPS": -1, "V1-LO": -1}
contrast = sum(wts[k] * asin(arr) for k, arr in right.items())
t_c, p_c = stats.ttest_1samp(contrast, 0.0)
dz_c = contrast.mean() / contrast.std(ddof=1)

print(f"\n4-level RM-ANOVA (for reference):")
print(aovR.anova_table)
print(f"Group contrast object(+1) vs control(-1): t({n-1}) = {t_c:.2f}, p = {p_c:.2e}, dz = {dz_c:.2f}")

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
for c, t, ph, dz in zip(wc, wt, wp_holm, wdz):
    print(f"  {c:22s} t={t:8.2f}  p_holm={ph:11.2e}  dz={dz:7.2f}")

print("\nPair means (Dice):")
for k, arr in {**left, **right}.items():
    print(f"  {k:16s}: {np.nanmean(arr):.3f}")


# ================================================================
# FIGURE 1: aCompCor robustness (Fig 3D replication)
# ================================================================
fig1, ax1 = plt.subplots(figsize=(5, 4.5))

bars1 = [("between-subj\ndorsal", bt_dorsal, TEAL),
         ("between-subj\nventral", bt_ventral, PINK),
         ("within-subj\ndorsal–ventral", w_pips_lo, PURPLE)]
for i, (lab, v, c) in enumerate(bars1):
    bar_with_dots(ax1, i, v, c)

ax1.set_xticks(range(len(bars1)))
ax1.set_xticklabels([l for l, _, _ in bars1], fontsize=9)
ax1.set_title("PPI network overlap (aCompCor)", fontsize=11)
sig_bar(ax1, 1, 2, 1.02, stars(pL_holm[1]))
sig_bar(ax1, 0, 2, 1.14, stars(pL_holm[0]))
style_ax(ax1, ylim_top=1.3)
fig1.tight_layout()

fig1_path = f"{out_dir}/fig_acompcor_robustness.png"
fig1.savefig(fig1_path, dpi=300, bbox_inches="tight")
print(f"\nSaved: {fig1_path}")


# ================================================================
# SUPP FIGURE: pFS control
# ================================================================
fig_pfs, ax_pfs = plt.subplots(figsize=(5, 4.5))

bar_with_dots(ax_pfs, 0, w_pips_lo, PURPLE)
bar_with_dots(ax_pfs, 1, w_pfs_pips, PURPLE)
bar_with_dots(ax_pfs, 2, w_pfs_lo, PURPLE)

ax_pfs.set_xticks([0, 1, 2])
ax_pfs.set_xticklabels(["pIPS – LO\n(primary)", "pFS – pIPS", "pFS – LO"], fontsize=10)
ax_pfs.set_title("Within-subject PPI network overlap: pFS", fontsize=11)
style_ax(ax_pfs)
fig_pfs.tight_layout()

fig_pfs_path = f"{out_dir}/supp_fig_pFS.png"
fig_pfs.savefig(fig_pfs_path, dpi=300, bbox_inches="tight")
print(f"Saved: {fig_pfs_path}")


# ================================================================
# SUPP FIGURE: V1 control
# ================================================================
fig_v1, ax_v1 = plt.subplots(figsize=(5, 4.5))

bar_with_dots(ax_v1, 0, w_pips_lo, PURPLE)
bar_with_dots(ax_v1, 1, w_v1_pips, GRAY)
bar_with_dots(ax_v1, 2, w_v1_lo, GRAY)

ax_v1.set_xticks([0, 1, 2])
ax_v1.set_xticklabels(["pIPS – LO\n(primary)", "V1 – pIPS", "V1 – LO"], fontsize=10)
ax_v1.set_title("Within-subject PPI network overlap: V1", fontsize=11)
style_ax(ax_v1)
fig_v1.tight_layout()

fig_v1_path = f"{out_dir}/supp_fig_V1.png"
fig_v1.savefig(fig_v1_path, dpi=300, bbox_inches="tight")
print(f"Saved: {fig_v1_path}")


# ---- save stats CSVs ----
aovL.anova_table.to_csv(f"{out_dir}/dice_anova_robustness.csv")
aovR.anova_table.to_csv(f"{out_dir}/dice_anova_4level.csv")
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
print("\nSaved all CSVs.")
