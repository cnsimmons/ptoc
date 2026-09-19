#!/usr/bin/env python3
"""
Communications Biology Source Data workbook — CommsBio-26-2997.
One .xlsx tab per graph/chart panel. Run: pip install openpyxl first.
"""
import os, sys
import pandas as pd
import numpy as np

BASE = "/user_data/csimmon2/git_repos/ptoc/results"
OUT_XLSX = f"{BASE}/source_data.xlsx"

# ── confirmed paths ──
P = {
    "fc_dice":       f"{BASE}/dice_figures/fc_dice_per_subject.csv",
    "partial_dice":  f"{BASE}/dice_figures/partial_dice_per_subject.csv",
    "fc_fp":         f"{BASE}/connectivity_comparison/bilateral_fc_connectivity_fingerprint_results_pearsonr.csv",
    "ppi_fp":        f"{BASE}/connectivity_comparison/bilateral_ppi_connectivity_fingerprint_results_pearsonr.csv",
    "partial_fp":    f"{BASE}/connectivity_comparison/bilateral_partial_correlation_connectivity_fingerprint_results_pearsonr.csv",
    "ppi_within":    f"{BASE}/dice_comparison/ppi_between_roi_dice_by_subject.csv",
    "ppi_btw_dors":  f"{BASE}/dice_comparison/pIPS_ppi_between_subject_dice.csv",
    "ppi_btw_vent":  f"{BASE}/dice_comparison/LO_ppi_between_subject_dice.csv",
    "acomp_orig_vs": f"{BASE}/acompcor_comparison/dice_original_vs_acompcor.csv",
    "acomp_persub":  f"{BASE}/acompcor_comparison/dice_per_subject.csv",
    "ctrl_pairs":    f"{BASE}/acompcor_comparison/dice_control_pairs.csv",
}

def load(key):
    p = P[key]
    if not os.path.exists(p):
        print(f"  MISSING: {p}")
        return None
    return pd.read_csv(p)

# ── builders ──

def build_ppi_dice():
    """Fig 3d: merge 3 PPI split files into one per-subject table."""
    within = load("ppi_within")
    btw_d  = load("ppi_btw_dors")
    btw_v  = load("ppi_btw_vent")
    if any(x is None for x in (within, btw_d, btw_v)):
        return None
    # within: already 1 row per subject
    w = within.rename(columns={"Subject": "subject", "Dice": "within"})[["subject", "within"]]
    # between: 153 pairwise rows → average per subject (appears as Subject1 and Subject2)
    def avg_per_sub(df):
        a = df.groupby("Subject1")["Dice"].mean()
        b = df.groupby("Subject2")["Dice"].mean()
        combined = pd.concat([a, b]).groupby(level=0).mean()
        return combined
    bd = avg_per_sub(btw_d).rename("between_dorsal")
    bv = avg_per_sub(btw_v).rename("between_ventral")
    out = w.set_index("subject").join(bd).join(bv).reset_index()
    return out

def build_supp_v1():
    """Supp Fig 3: V1-pIPS, V1-LO, with pIPS-LO reference."""
    cp = load("ctrl_pairs")
    ps = load("acomp_persub")
    if cp is None or ps is None:
        return None
    return pd.DataFrame({
        "subject": ps["subject"],
        "pIPS_LO": ps["pIPS_LO"],
        "V1_pIPS": cp["V1_pIPS"],
        "V1_LO":   cp["V1_LO"],
    })

def build_supp_pfs():
    """Supp Fig 4: pFS-pIPS, pFS-LO, with pIPS-LO reference."""
    cp = load("ctrl_pairs")
    ps = load("acomp_persub")
    if cp is None or ps is None:
        return None
    return pd.DataFrame({
        "subject":  ps["subject"],
        "pIPS_LO":  ps["pIPS_LO"],
        "PFS_pIPS": cp["PFS_pIPS"],
        "PFS_LO":   cp["PFS_LO"],
    })

def g_table(both, dors_only, vent_only, neither):
    """G panels: parcel proportions from thresholded group maps (not in CSVs)."""
    total = both + dors_only + vent_only + neither
    return pd.DataFrame({
        "category":  ["both", "dorsal_only", "ventral_only", "neither"],
        "n_parcels": [both, dors_only, vent_only, neither],
        "percent":   [round(100*x/total, 1) for x in [both, dors_only, vent_only, neither]],
        "total_parcels": [total]*4,
    })

def fp_select(key, cols):
    df = load(key)
    if df is None:
        return None
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"  WARNING: cols {missing} not in {key}, writing full table")
        return df
    out = df[cols].copy()
    # Write booleans as plain text so Excel does not store them as =TRUE()/=FALSE()
    if "Combined_Significant" in out.columns:
        out["Combined_Significant"] = (
            out["Combined_Significant"].astype(bool)
            .map({True: "TRUE", False: "FALSE"})
        )
    return out

def build_readme():
    """README worksheet: worksheet index plus column definitions."""
    L = [
        "Communications Biology Source Data - CommsBio-26-2997",
        "Large-scale functional overlap between dorsal and ventral object-responsive networks",
        "Simmons C, Behrmann M, Ayzenberg V",
        "",
        "One worksheet per graph/chart panel. Values are exactly as plotted.",
        "Raw data deposit: https://doi.org/10.1184/R1/31459006",
        "Analysis code: https://github.com/cnsimmons/dv_overlap",
        "",
        "WORKSHEET INDEX",
        "  Fig 2d       Dice coefficients, correlation-based functional connectivity",
        "  Fig 2e       Parcel connectivity strength (Pearson's r), dorsal and ventral seeds",
        "  Fig 2f       Dorsal minus ventral difference per parcel, with significance flag",
        "  Fig 2g       Parcel proportions by connectivity category",
        "  Fig 3d       Dice coefficients, task-dependent connectivity (PPI)",
        "  Fig 3e       Parcel connectivity strength (PPI)",
        "  Fig 3f       Dorsal minus ventral difference per parcel (PPI)",
        "  Fig 3g       Parcel proportions by connectivity category (PPI)",
        "  Fig 4d       Dice coefficients, partial correlation",
        "  Fig 4e       Parcel connectivity strength (partial correlation)",
        "  Fig 4f       Dorsal minus ventral difference per parcel (partial correlation)",
        "  Fig 4g       Parcel proportions by connectivity category (partial correlation)",
        "  Supp Fig 2   Dice overlap, original versus aCompCor pipelines",
        "  Supp Fig 3   Dice overlap, V1 control seed",
        "  Supp Fig 4   Dice overlap, pFS control seed",
        "",
        "Manuscript Supplementary Figure 1 is an anatomical parcel image and has no",
        "plotted values, so it has no worksheet here.",
        "",
        "COLUMN DEFINITIONS",
        "  subject                One row per participant (n = 18 per experiment).",
        "  within                 Within-participant Dice between the dorsal (pIPS) and",
        "                         ventral (LO) whole-brain network maps.",
        "  between_dorsal         Mean Dice between this participant's dorsal map and every",
        "                         other participant's dorsal map.",
        "  between_ventral        As above, for the ventral map.",
        "  pIPS_LO                Within-participant dorsal-ventral Dice (reference bar in",
        "                         the supplementary control panels).",
        "  V1_pIPS, V1_LO         Within-participant Dice between the V1 control seed and the",
        "                         dorsal or ventral seed network.",
        "  PFS_pIPS, PFS_LO       As above, for the pFS seed.",
        "  ROI_Name               Parcel label from the merged Schaefer-Wang-Julian atlas.",
        "  pIPS_Connectivity      Connectivity between the dorsal seed and that parcel,",
        "                         as Pearson's r.",
        "  LO_Connectivity        As above, for the ventral seed.",
        "  Difference             pIPS_Connectivity minus LO_Connectivity.",
        "  Combined_Significant   TRUE if the difference was significant against bootstrapped",
        "                         confidence intervals; plotted as an opaque bar.",
        "  category               both / dorsal_only / ventral_only / neither.",
        "  n_parcels, percent     Parcel counts and percentages in each category.",
        "  total_parcels          Total parcels considered (200).",
        "",
        "NOTES",
        "  Dice coefficients are arcsine-square-root transformed before statistical",
        "  analysis; the values here are untransformed, as plotted.",
        "  Bars show the mean across participants with 95% confidence intervals.",
        "  G-panel proportions derive from thresholded group maps rather than any single",
        "  per-subject file; counts are taken from the manuscript text.",
        "  Surface and statistical maps are covered by the data deposit above.",
    ]
    return pd.DataFrame({0: L})


# ── assembly ──

def build_all():
    sheets = {}
    notes = []
    def add(name, df):
        if df is not None:
            sheets[name] = df
        else:
            notes.append(f"{name}: SKIPPED")

    # Fig 2
    add("Fig 2d", load("fc_dice"))
    add("Fig 2e", fp_select("fc_fp", ["ROI_Name","pIPS_Connectivity","LO_Connectivity"]))
    add("Fig 2f", fp_select("fc_fp", ["ROI_Name","Difference","Combined_Significant"]))
    add("Fig 2g", g_table(92, 33, 3, 72))   # from MS: 92 of 128 connected; 33 unique dorsal; 3 unique ventral

    # Fig 3
    add("Fig 3d", build_ppi_dice())
    add("Fig 3e", fp_select("ppi_fp", ["ROI_Name","pIPS_Connectivity","LO_Connectivity"]))
    add("Fig 3f", fp_select("ppi_fp", ["ROI_Name","Difference","Combined_Significant"]))
    add("Fig 3g", g_table(65, 50, 3, 82))   # from MS: 65 of 118 connected; 50 unique dorsal; 3 unique ventral

    # Fig 4
    add("Fig 4d", load("partial_dice"))
    add("Fig 4e", fp_select("partial_fp", ["ROI_Name","pIPS_Connectivity","LO_Connectivity"]))
    add("Fig 4f", fp_select("partial_fp", ["ROI_Name","Difference","Combined_Significant"]))
    add("Fig 4g", g_table(55, 61, 19, 65))  # from MS: 55 both; 61 unique dorsal; 19 unique ventral

    # Supp
    # Manuscript Supplementary Figure 1 is the anatomical parcel image and has no
    # plotted values, so the workbook tabs start at Supp Fig 2.
    add("Supp Fig 2", load("acomp_orig_vs"))   # aCompCor vs original
    add("Supp Fig 3", build_supp_v1())         # V1 control
    add("Supp Fig 4", build_supp_pfs())        # pFS control

    return sheets, notes

def inspect():
    """Quick sanity check printed before writing."""
    print("=" * 60)
    print("SANITY CHECKS")
    print("=" * 60)
    for tag, tgt in [("fc_dice",(0.914,0.830,0.789)), ("partial_dice",(0.706,0.786,0.687))]:
        df = load(tag)
        if df is not None:
            m = df.select_dtypes("number").mean()
            print(f"[{tag}] means: {', '.join(f'{c}={m[c]:.3f}' for c in m.index)}")
            print(f"  manuscript: {tgt}")
    # PPI dice
    ppi = build_ppi_dice()
    if ppi is not None:
        m = ppi.select_dtypes("number").mean()
        print(f"[ppi_dice] means: {', '.join(f'{c}={m[c]:.3f}' for c in m.index)}")
        print(f"  manuscript: (0.810, 0.525, 0.535)")
    # fingerprint scale
    for tag in ("fc_fp","ppi_fp","partial_fp"):
        df = load(tag)
        if df is not None and "pIPS_Connectivity" in df.columns:
            print(f"[{tag}] pIPS mean = {df['pIPS_Connectivity'].mean():.4f}")

def main():
    inspect()
    sheets, notes = build_all()
    if not sheets:
        print("\nNothing built."); sys.exit(1)

    readme = build_readme()
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as xw:
        # header=False so the README reads as prose, not a table with a column header
        readme.to_excel(xw, sheet_name="README", index=False, header=False)
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name[:31], index=False)

    print(f"\nWROTE {OUT_XLSX}  ({len(sheets)} tabs + README)")
    print("Tabs: " + ", ".join(sheets.keys()))
    if notes:
        print("\nCHECK THESE:")
        for n in notes:
            print("  - " + n)

if __name__ == "__main__":
    main()
