#!/usr/bin/env python3
"""
Partial Correlation Strength Comparison for PTOC Paper
======================================================
Computes dorsal vs ventral connectivity strength within the overlapping
parcels from the partial correlation analysis.

Adapted from the FC/PPI bootstrap+LOO pipeline.
Uses the "cleaned" (residualized) maps: *_clean_*_fc_mni.nii.gz

Output: Count of parcels where dorsal > ventral vs ventral > dorsal
        among the overlapping parcels, plus chi-square test.
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.maskers import NiftiLabelsMasker
from nilearn import image
from sklearn.utils import resample
from scipy import stats, ndimage

# =============================================================================
# CONFIGURATION — UPDATE THESE PATHS AS NEEDED
# =============================================================================
study_dir = "/lab_data/behrmannlab/vlad/ptoc"
sub_info_path = '/user_data/csimmon2/git_repos/ptoc/sub_info.csv'
results_dir = '/user_data/csimmon2/git_repos/ptoc/results'
output_dir = f'{results_dir}/connectivity_comparison'
os.makedirs(output_dir, exist_ok=True)

# Path to merged atlas
atlas_path = f'{results_dir}/schaefer_wang_merged.nii.gz'
merged_labels_file = f'{results_dir}/merged_atlas_labels.npy'

# Path to partial correlation (cleaned) maps
# UPDATE this if your residual maps are stored elsewhere
residual_dir_template = f'{study_dir}/{{sub}}/ses-01/derivatives'
# Expected naming: {sub}_pIPS_clean_{hemi}_loc_fc_mni.nii.gz
#                  {sub}_LO_clean_{hemi}_loc_fc_mni.nii.gz

# Two possible locations for the MNI-transformed cleaned maps:
# Option A: Same derivatives folder as regular FC
PARTIAL_MAP_DIRS = [
    '/user_data/csimmon2/ptoc_residuals/{sub}/ses-01/derivatives/fc_mni',
]

MIN_VOXELS = 5  # Manuscript cluster threshold for parcel classification

# =============================================================================
# LOAD ATLAS
# =============================================================================
print("Loading atlas...")
atlas_img = nib.load(atlas_path)
atlas_data = atlas_img.get_fdata()
atlas_labels = np.load(merged_labels_file, allow_pickle=True)
all_labels = np.unique(atlas_data)
all_labels = all_labels[all_labels > 0]

# Get hemisphere and size for each parcel
atlas_affine = atlas_img.affine
hemi_dict = {}
size_dict = {}
centers = ndimage.center_of_mass(atlas_data, labels=atlas_data, index=all_labels)
for label, center in zip(all_labels, centers):
    vox_coord = np.append(center, 1)
    world_coord = np.dot(atlas_affine, vox_coord)
    hemi_dict[int(label)] = 'left' if world_coord[0] < 0 else 'right'
    size_dict[int(label)] = np.sum(atlas_data == label)

masker = NiftiLabelsMasker(labels_img=atlas_path, resampling_target='data')
n_parcels = len(all_labels)
print(f"Atlas: {n_parcels} parcels")

# =============================================================================
# IDENTIFY THE 53 OVERLAPPING PARCELS FROM PARTIAL CORRELATION ANALYSIS
# =============================================================================
print("\nIdentifying overlapping parcels from partial correlation maps...")

# Load the group-level thresholded partial correlation maps
# These should be the FDR-corrected group maps
# UPDATE these paths if they differ in your setup
partial_thresh_paths = {
    'L_Dorsal': f'{results_dir}/partial_correlation_thresh_pIPS/pIPS_left_cleaned_thresh.nii.gz',
    'L_Ventral': f'{results_dir}/partial_correlation_thresh_LO/LO_left_cleaned_thresh.nii.gz',
    'R_Dorsal': f'{results_dir}/partial_correlation_thresh_pIPS/pIPS_right_cleaned_thresh.nii.gz',
    'R_Ventral': f'{results_dir}/partial_correlation_thresh_LO/LO_right_cleaned_thresh.nii.gz',
}

# Check which paths exist; try alternative naming if needed
alt_partial_paths = {
    'L_Dorsal': f'{results_dir}/group_averages/pIPS_clean_left_fc_thresh.nii.gz',
    'L_Ventral': f'{results_dir}/group_averages/LO_clean_left_fc_thresh.nii.gz',
    'R_Dorsal': f'{results_dir}/group_averages/pIPS_clean_right_fc_thresh.nii.gz',
    'R_Ventral': f'{results_dir}/group_averages/LO_clean_right_fc_thresh.nii.gz',
}

# Try primary paths, fall back to alt
for key in partial_thresh_paths:
    if not os.path.exists(partial_thresh_paths[key]):
        if os.path.exists(alt_partial_paths[key]):
            partial_thresh_paths[key] = alt_partial_paths[key]
            print(f"  Using alt path for {key}: {alt_partial_paths[key]}")
        else:
            print(f"  WARNING: Cannot find threshold map for {key}")
            print(f"    Tried: {partial_thresh_paths[key]}")
            print(f"    Tried: {alt_partial_paths[key]}")

# Classify parcels using the partial correlation threshold maps
counts_per_map = {}
sorted_labels = np.sort(all_labels)

for key, path in partial_thresh_paths.items():
    if os.path.exists(path):
        binary_img = image.math_img("img > 0", img=path)
        proportions = masker.fit_transform(binary_img).flatten()
        sizes = np.array([size_dict[lbl] for lbl in sorted_labels])
        counts_per_map[key] = proportions * sizes
    else:
        print(f"  MISSING: {path} — setting all to 0")
        counts_per_map[key] = np.zeros(len(sorted_labels))

# Identify overlapping parcels
overlap_parcel_indices = []
dorsal_only_indices = []
ventral_only_indices = []

for i, label in enumerate(sorted_labels):
    hemi = hemi_dict[label]
    if hemi == 'left':
        is_dorsal = counts_per_map['L_Dorsal'][i] >= MIN_VOXELS
        is_ventral = counts_per_map['L_Ventral'][i] >= MIN_VOXELS
    else:
        is_dorsal = counts_per_map['R_Dorsal'][i] >= MIN_VOXELS
        is_ventral = counts_per_map['R_Ventral'][i] >= MIN_VOXELS
    
    if is_dorsal and is_ventral:
        overlap_parcel_indices.append(i)
    elif is_dorsal:
        dorsal_only_indices.append(i)
    elif is_ventral:
        ventral_only_indices.append(i)

n_overlap = len(overlap_parcel_indices)
n_dorsal_only = len(dorsal_only_indices)
n_ventral_only = len(ventral_only_indices)
print(f"\nPartial correlation parcel classification:")
print(f"  Overlapping: {n_overlap}")
print(f"  Dorsal only: {n_dorsal_only}")
print(f"  Ventral only: {n_ventral_only}")
print(f"  Neither: {n_parcels - n_overlap - n_dorsal_only - n_ventral_only}")

if n_overlap == 0:
    print("\nERROR: No overlapping parcels found. Check threshold map paths.")
    print("Listing files in group_averages to help debug:")
    ga_dir = f'{results_dir}/group_averages'
    if os.path.exists(ga_dir):
        for f in sorted(os.listdir(ga_dir)):
            if 'clean' in f.lower() or 'partial' in f.lower() or 'resid' in f.lower():
                print(f"  {f}")
    exit(1)

# =============================================================================
# LOAD SUBJECT-LEVEL PARTIAL CORRELATION MAPS
# =============================================================================
print("\nLoading subject-level partial correlation maps...")

sub_info = pd.read_csv(sub_info_path)
subjects = sub_info[sub_info['group'] == 'control']['sub'].tolist()
if 'sub-084' in subjects:
    subjects.remove('sub-084')
    print("Excluded sub-084")

hemispheres = ['left', 'right']

def find_partial_map(sub, roi, hemi):
    """Try to find the partial/cleaned FC map for a given subject, ROI, hemisphere."""
    # Try several naming conventions
    patterns = [
        # Pattern 1: *_clean_*_loc_fc_mni.nii.gz
        f'{sub}_{roi}_clean_{hemi}_loc_fc_mni.nii.gz',
        # Pattern 2: *_clean_*_fc_mni.nii.gz (no "loc")
        f'{sub}_{roi}_clean_{hemi}_fc_mni.nii.gz',
        # Pattern 3: *_{roi}_{hemi}_partial_mni.nii.gz
        f'{sub}_{roi}_{hemi}_loc_partial_mni.nii.gz',
    ]
    
    for dir_template in PARTIAL_MAP_DIRS:
        search_dir = dir_template.format(sub=sub)
        if os.path.exists(search_dir):
            for pattern in patterns:
                full_path = os.path.join(search_dir, pattern)
                if os.path.exists(full_path):
                    return full_path
    return None

subject_data = []
missing_subjects = []

for sub in subjects:
    sub_conn = {}
    
    for roi in ['pIPS', 'LO']:
        combined_data = None
        hemi_count = 0
        
        for hemi in hemispheres:
            fpath = find_partial_map(sub, roi, hemi)
            
            if fpath is not None:
                try:
                    fc_img = nib.load(fpath)
                    fc_values = masker.fit_transform(fc_img)[0]
                    
                    if combined_data is None:
                        combined_data = fc_values
                    else:
                        combined_data += fc_values
                    hemi_count += 1
                except Exception as e:
                    print(f"  Error processing {fpath}: {e}")
            # else: file not found for this hemisphere
        
        if hemi_count > 0:
            sub_conn[roi] = combined_data / hemi_count
    
    if len(sub_conn) == 2:
        subject_data.append(sub_conn)
    else:
        missing_subjects.append(sub)

n_valid = len(subject_data)
print(f"\nLoaded partial correlation data for {n_valid} / {len(subjects)} subjects")
if missing_subjects:
    print(f"  Missing subjects: {missing_subjects[:5]}{'...' if len(missing_subjects) > 5 else ''}")

if n_valid < 3:
    print("\nERROR: Not enough subjects with partial correlation maps.")
    print("Trying to list available files for first subject to debug naming...")
    test_sub = subjects[0]
    for dir_template in PARTIAL_MAP_DIRS:
        d = dir_template.format(sub=test_sub)
        if os.path.exists(d):
            print(f"\nFiles in {d} containing 'clean' or 'partial':")
            for f in sorted(os.listdir(d)):
                if 'clean' in f.lower() or 'partial' in f.lower():
                    print(f"  {f}")
    exit(1)

# =============================================================================
# EXTRACT CONNECTIVITY VALUES FOR OVERLAPPING PARCELS
# =============================================================================
print(f"\nAnalyzing strength differences in {n_overlap} overlapping parcels...")

n_rois = len(sorted_labels)
pips_vals = np.zeros((n_valid, n_rois))
lo_vals = np.zeros((n_valid, n_rois))

for i, sub_conn in enumerate(subject_data):
    pips_vals[i, :] = sub_conn['pIPS']
    lo_vals[i, :] = sub_conn['LO']

# Restrict to overlapping parcels
overlap_idx = np.array(overlap_parcel_indices)
pips_overlap = pips_vals[:, overlap_idx]
lo_overlap = lo_vals[:, overlap_idx]

# Mean across subjects
mean_pips = np.mean(pips_overlap, axis=0)
mean_lo = np.mean(lo_overlap, axis=0)
diff_profile = mean_pips - mean_lo

# =============================================================================
# BOOTSTRAP ANALYSIS
# =============================================================================
print("Running bootstrap analysis (10,000 iterations)...")
n_boots = 10000
boot_diffs = np.zeros((n_boots, n_overlap))

for b in range(n_boots):
    boot_idx = resample(range(n_valid), replace=True, n_samples=n_valid)
    boot_pips = np.mean(pips_overlap[boot_idx, :], axis=0)
    boot_lo = np.mean(lo_overlap[boot_idx, :], axis=0)
    boot_diffs[b, :] = boot_pips - boot_lo

ci_lower = np.percentile(boot_diffs, 2.5, axis=0)
ci_upper = np.percentile(boot_diffs, 97.5, axis=0)
sig_boot = (ci_lower > 0) | (ci_upper < 0)

# =============================================================================
# LOO ANALYSIS
# =============================================================================
print("Running LOO cross-validation...")
loo_reliability = np.zeros(n_overlap)

for left_out in range(n_valid):
    train_idx = [j for j in range(n_valid) if j != left_out]
    train_diff = np.mean(pips_overlap[train_idx, :] - lo_overlap[train_idx, :], axis=0)
    test_diff = pips_overlap[left_out, :] - lo_overlap[left_out, :]
    loo_reliability += (np.sign(train_diff) == np.sign(test_diff)).astype(float)

loo_reliability /= n_valid
loo_threshold = 0.75
sig_loo = loo_reliability >= loo_threshold

# Combined significance
sig_combined = sig_boot & sig_loo

# =============================================================================
# COUNT AND CHI-SQUARE
# =============================================================================
dorsal_stronger = np.sum(sig_combined & (diff_profile > 0))
ventral_stronger = np.sum(sig_combined & (diff_profile < 0))
not_sig = n_overlap - dorsal_stronger - ventral_stronger

print(f"\n{'='*60}")
print(f"RESULTS: Partial Correlation Strength in {n_overlap} Overlapping Parcels")
print(f"{'='*60}")
print(f"  Dorsal (pIPS) significantly stronger: {dorsal_stronger}")
print(f"  Ventral (LO) significantly stronger:  {ventral_stronger}")
print(f"  Not significantly different:           {not_sig}")

# Chi-square on dorsal vs ventral counts
if dorsal_stronger + ventral_stronger > 0:
    observed = np.array([dorsal_stronger, ventral_stronger])
    expected = np.array([(dorsal_stronger + ventral_stronger) / 2] * 2)
    chi2 = np.sum((observed - expected)**2 / expected)
    p_val = 1 - stats.chi2.cdf(chi2, df=1)
    print(f"\n  χ²(1) = {chi2:.2f}, p = {p_val:.4f}")
    if p_val < .001:
        print(f"  → Significant asymmetry (p < .001)")
    elif p_val < .05:
        print(f"  → Significant asymmetry (p < .05)")
    else:
        print(f"  → No significant asymmetry")
else:
    print("\n  No significant parcels found — cannot compute χ²")

# =============================================================================
# FOR THE MANUSCRIPT
# =============================================================================
print(f"\n{'='*60}")
print("TEXT FOR MANUSCRIPT (paste into placeholder):")
print(f"{'='*60}")
print(f"In terms of connectivity strength within the {n_overlap} overlapping")
print(f"parcels, the dorsal ROI exhibited significantly stronger connectivity")
print(f"than the ventral ROI in {dorsal_stronger} parcels, while the ventral ROI")
print(f"showed stronger connectivity in {ventral_stronger} parcels", end="")
if dorsal_stronger + ventral_stronger > 0:
    print(f" (χ²(1) = {chi2:.2f}, p {'< .001' if p_val < .001 else '= ' + f'{p_val:.3f}'}; Figure 4E-F).")
else:
    print(" (Figure 4E-F).")

# =============================================================================
# SAVE DETAILED RESULTS
# =============================================================================
# Get labels for overlap parcels
overlap_labels_list = sorted_labels[overlap_idx]

def get_label_name(label_val):
    try:
        idx = int(label_val) - 1
        if 0 <= idx < len(atlas_labels):
            lab = atlas_labels[idx]
            if isinstance(lab, bytes):
                lab = lab.decode('utf-8')
            return lab
    except:
        pass
    return f"ROI_{label_val}"

results_df = pd.DataFrame({
    'Parcel_Label': overlap_labels_list,
    'Parcel_Name': [get_label_name(l) for l in overlap_labels_list],
    'pIPS_Mean_Connectivity': mean_pips,
    'LO_Mean_Connectivity': mean_lo,
    'Difference_pIPS_minus_LO': diff_profile,
    'CI_Lower': ci_lower,
    'CI_Upper': ci_upper,
    'Bootstrap_Significant': sig_boot,
    'LOO_Reliability': loo_reliability,
    'LOO_Significant': sig_loo,
    'Combined_Significant': sig_combined,
    'Direction': np.where(diff_profile > 0, 'pIPS > LO', 'LO > pIPS')
})

csv_path = f'{output_dir}/partial_correlation_overlap_strength_comparison.csv'
results_df.to_csv(csv_path, index=False)
print(f"\nDetailed results saved to: {csv_path}")