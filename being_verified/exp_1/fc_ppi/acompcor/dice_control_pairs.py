"""
R2.2 (V1 control) + R2.3 (pFS): within-subject Dice on aCompCor PPI maps.
Pairs: PFS-pIPS (object-object, R2.3); V1-pIPS + V1-LO (control, R2.2).
Method identical to dice_original_vs_acompcor: hemisphere-averaged, >0 binarize.
Reference: existing pIPS-LO aCompCor PPI Dice = 0.787.
"""
import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import stats

study_dir = "/lab_data/behrmannlab/vlad/ptoc"
results_dir = "/user_data/csimmon2/git_repos/ptoc/results"

sub_info = pd.read_csv("/user_data/csimmon2/git_repos/ptoc/sub_info.csv")
subs = sub_info[sub_info["group"] == "control"]["sub"].tolist()
subs = [s for s in subs if s != "sub-084"]      # documented exclusion (N=18)

rois = ["pIPS", "LO", "PFS", "V1"]
hemispheres = ["left", "right"]
pairs = [("PFS", "pIPS"), ("V1", "pIPS"), ("V1", "LO"), ("PFS", "LO")]


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


data, valid = load_maps(subs)
print(f"\nValid subjects (all 4 ROIs, both hemis): {len(valid)}")

rows = []
for sub in valid:
    row = {"subject": sub}
    for a, b in pairs:
        row[f"{a}_{b}"] = dice(data[sub][a], data[sub][b])
    rows.append(row)
df = pd.DataFrame(rows)

print(f"\n{'pair':12s}{'mean':>9s}{'95% CI':>22s}")
for a, b in pairs:
    v = df[f"{a}_{b}"].dropna().values
    m = v.mean()
    lo, hi = stats.t.interval(0.95, len(v) - 1, loc=m, scale=stats.sem(v))
    print(f"{a+'-'+b:12s}{m:9.4f}   [{lo:.4f}, {hi:.4f}]  n={len(v)}")

out = f"{results_dir}/acompcor_comparison"
os.makedirs(out, exist_ok=True)
df.to_csv(f"{out}/dice_control_pairs.csv", index=False)
print(f"\nSaved: {out}/dice_control_pairs.csv")
