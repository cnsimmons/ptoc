# Regional Brain Connectivity Analysis Part 1: Analysis and CSV Export
# Description: Analysis of functional connectivity (FC) and psychophysiological interaction (PPI)
#              between parietal IPS and lateral occipital regions with CSV export

import os
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker
from sklearn.utils import resample

# Define study paths
study_dir = "/lab_data/behrmannlab/vlad/ptoc"
sub_info_path = '/user_data/csimmon2/git_repos/ptoc/sub_info.csv'
results_dir = '/user_data/csimmon2/git_repos/ptoc/results'
output_dir = f'{results_dir}/connectivity_comparison'
os.makedirs(output_dir, exist_ok=True)

def analyze_connectivity_and_save_results(analysis_type='fc'):
    """Analyze connectivity and save results as CSV using merged atlas"""
    print(f"\n{'='*80}")
    print(f"ANALYZING {analysis_type.upper()} CONNECTIVITY AND SAVING RESULTS")
    print(f"{'='*80}")
    
    # Load merged atlas instead of fetching standard Schaefer atlas
    merged_atlas_file = f'{results_dir}/schaefer_wang_merged.nii.gz'
    merged_labels_file = f'{results_dir}/merged_atlas_labels.npy'
    
    if not os.path.exists(merged_atlas_file) or not os.path.exists(merged_labels_file):
        print("Error: Merged atlas files not found. Please run merge_atlas.py first.")
        return None
    
    atlas_img = nib.load(merged_atlas_file)
    atlas_labels = np.load(merged_labels_file, allow_pickle=True)
    
    print(f"Loaded merged atlas with {len(atlas_labels)} parcels")
    
    # Load subject info
    sub_info = pd.read_csv(sub_info_path)
    subjects = sub_info[sub_info['group'] == 'control']['sub'].tolist()
    
    # Exclude sub-084 as specified
    if 'sub-084' in subjects:
        subjects.remove('sub-084')
        print("Excluded sub-084 from analysis")
        
    print(f"Found {len(subjects)} control subjects")
    
    # Define ROIs and hemispheres
    rois = ['pIPS', 'LO']
    hemispheres = ['left', 'right']
    
    # Setup atlas masker
    masker = NiftiLabelsMasker(labels_img=atlas_img, standardize=False)
    
    # Load and process subject data
    subject_data = []
    for sub in subjects:
        sub_conn = {}
        
        for roi in rois:
            # Initialize arrays to hold combined data
            combined_data = None
            hemi_count = 0
            
            for hemisphere in hemispheres:
                fc_file = f'{study_dir}/{sub}/ses-01/derivatives/fc_mni/{sub}_{roi}_{hemisphere}_loc_{analysis_type}_mni.nii.gz'
                
                if os.path.exists(fc_file):
                    try:
                        # Load the FC map
                        fc_img = nib.load(fc_file)
                        
                        # Extract ROI values using atlas
                        fc_values = masker.fit_transform(fc_img)[0]
                        
                        # Add to combined data
                        if combined_data is None:
                            combined_data = fc_values
                        else:
                            combined_data += fc_values
                        
                        hemi_count += 1
                    except Exception as e:
                        print(f"Error processing {fc_file}: {e}")
            
            # Average the data if we have at least one hemisphere
            if hemi_count > 0:
                sub_conn[roi] = combined_data / hemi_count
        
        # Only include subjects with both ROIs
        if len(sub_conn) == len(rois):
            subject_data.append(sub_conn)
    
    n_valid_subjects = len(subject_data)
    print(f"Successfully loaded data for {n_valid_subjects} subjects")
    
    if n_valid_subjects < 3:
        print("Not enough valid subjects. Analysis cannot proceed.")
        return None
    
    # Initialize arrays for analysis
    n_rois = len(atlas_labels)
    pips_fc = np.zeros((n_valid_subjects, n_rois))
    lo_fc = np.zeros((n_valid_subjects, n_rois))
    
    # Fill arrays with connectivity data
    for i, subj_data in enumerate(subject_data):
        pips_fc[i, :] = subj_data['pIPS']
        lo_fc[i, :] = subj_data['LO']
    
    # Calculate connectivity vector correlation for each subject
    vector_correlations = np.zeros(n_valid_subjects)
    for i in range(n_valid_subjects):
        vector_correlations[i] = np.corrcoef(pips_fc[i, :], lo_fc[i, :])[0, 1]
    
    print(f"Mean connectivity vector correlation: {np.mean(vector_correlations):.3f} ± {np.std(vector_correlations):.3f}")
    
    # Calculate mean connectivity profiles
    mean_pips = np.mean(pips_fc, axis=0)
    mean_lo = np.mean(lo_fc, axis=0)
    
    # Remove self-connectivity
    # Get indices for Wang ROIs in the merged atlas
    wang_pips_idx = next((i for i, label in enumerate(atlas_labels) if 'Wang_pIPS' in str(label)), None)
    wang_lo_idx = next((i for i, label in enumerate(atlas_labels) if 'Wang_LO' in str(label)), None)
    
    # Print which indices are being excluded
    if wang_pips_idx is not None:
        print(f"Excluding Wang_pIPS (index {wang_pips_idx}) from connectivity analysis")
    if wang_lo_idx is not None:
        print(f"Excluding Wang_LO (index {wang_lo_idx}) from connectivity analysis")
    
    # Create mask for non-self connections (all True except at Wang ROI indices)
    mask = np.ones(len(atlas_labels), dtype=bool)
    if wang_pips_idx is not None:
        mask[wang_pips_idx] = False
    if wang_lo_idx is not None:
        mask[wang_lo_idx] = False
    
    # Apply mask to connectivity data
    pips_fc_masked = pips_fc[:, mask]
    lo_fc_masked = lo_fc[:, mask]
    
    # Get the masked atlas labels
    atlas_labels_masked = [label for i, label in enumerate(atlas_labels) if mask[i]]
    
    # Update mean connectivity with masked data
    mean_pips_masked = np.mean(pips_fc_masked, axis=0)
    mean_lo_masked = np.mean(lo_fc_masked, axis=0)
    
    # Get the new number of ROIs after masking
    n_masked_rois = np.sum(mask)
    print(f"Analyzing {n_masked_rois} ROIs after excluding Wang ROIs")
    
    # Calculate difference profile with masked data
    diff_profile = mean_pips_masked - mean_lo_masked
    abs_diff_profile = np.abs(diff_profile)
    
    # Run bootstrap analysis with masked data
    print("Performing bootstrap analysis...")
    n_boots = 10000
    boot_diffs = np.zeros((n_boots, n_masked_rois))
    
    for i in range(n_boots):
        # Resample subjects with replacement
        boot_idx = resample(range(n_valid_subjects), replace=True, n_samples=n_valid_subjects)
        
        # Calculate mean difference for this bootstrap sample
        boot_pips = np.mean(pips_fc_masked[boot_idx, :], axis=0)
        boot_lo = np.mean(lo_fc_masked[boot_idx, :], axis=0)
        boot_diffs[i, :] = boot_pips - boot_lo
    
    # Calculate confidence intervals
    ci_lower = np.percentile(boot_diffs, 2.5, axis=0)
    ci_upper = np.percentile(boot_diffs, 97.5, axis=0)
    
    # Identify significant differences (95% CI doesn't cross zero)
    sig_boot = (ci_lower > 0) | (ci_upper < 0)
    sig_boot_count = np.sum(sig_boot)
    print(f"Found {sig_boot_count} ROIs with significant differences via bootstrap")
    
    # Leave-One-Out Cross-Validation for reliability
    print("Performing Leave-One-Out Cross-Validation...")
    loo_reliability = np.zeros(n_masked_rois)
    
    for left_out in range(n_valid_subjects):
        # Create the training set (all subjects except the left out one)
        train_idx = list(range(n_valid_subjects))
        train_idx.remove(left_out)
        
        # Calculate mean training differences
        train_diff = np.mean(pips_fc_masked[train_idx, :] - lo_fc_masked[train_idx, :], axis=0)
        
        # Test on left out subject
        test_diff = pips_fc_masked[left_out, :] - lo_fc_masked[left_out, :]
        
        # Calculate consistency of signs between training and test
        loo_reliability += (np.sign(train_diff) == np.sign(test_diff)).astype(float)
    
    # Convert to proportion of consistent predictions
    loo_reliability = loo_reliability / n_valid_subjects
    
    # Set threshold at 75% consistency
    loo_threshold = 0.75
    sig_loo = loo_reliability >= loo_threshold
    sig_loo_count = np.sum(sig_loo)
    
    print(f"LOO reliability threshold: {loo_threshold:.3f}")
    print(f"Found {sig_loo_count} ROIs with high reliability via LOO")
    
    # Combine significance
    sig_combined = sig_boot & sig_loo
    sig_combined_count = np.sum(sig_combined)
    print(f"Found {sig_combined_count} ROIs significant with combined criteria")
    
    # Helper function to get ROI name
    def get_roi_name(roi_id):
        try:
            idx = int(roi_id) - 1
            if 0 <= idx < len(atlas_labels):
                label = atlas_labels[idx]
                if isinstance(label, bytes):
                    label = label.decode('utf-8')
                return label
        except:
            pass
        return f"ROI_{roi_id}"
    
    # Create a map from masked indices to original atlas indices
    original_indices = np.where(mask)[0]
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'ROI_ID': original_indices + 1,  # Convert to 1-based indexing
        'ROI_Name': [get_roi_name(i+1) for i in original_indices],
        'pIPS_Connectivity': mean_pips_masked,
        'LO_Connectivity': mean_lo_masked,
        'Difference': diff_profile,
        'Abs_Difference': abs_diff_profile,
        'CI_Lower': ci_lower,
        'CI_Upper': ci_upper,
        'LOO_Reliability': loo_reliability,
        'Bootstrap_Significant': sig_boot,
        'LOO_Significant': sig_loo,
        'Combined_Significant': sig_combined,
        'Direction': np.where(diff_profile > 0, 'pIPS > LO', 'LO > pIPS')
    })
    
    # Save results to CSV with "_rerun" suffix for comparison
    csv_path = f'{output_dir}/bilateral_{analysis_type}_connectivity_fingerprint_results_rerun.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    print(f"Compare this to: {output_dir}/bilateral_{analysis_type}_connectivity_fingerprint_results.csv")
    
    # Create roi_data dictionary for visualization
    roi_data = {
        'mean_pips': mean_pips_masked,
        'mean_lo': mean_lo_masked,
        'diff_profile': diff_profile,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'sig_combined': sig_combined
    }
    
    return results_df, roi_data

# Run analyses for both FC and PPI
print("\nRunning FC and PPI analyses and saving results...")
print("\n----- FC ANALYSIS -----")
fc_results, fc_data = analyze_connectivity_and_save_results(analysis_type='fc')
print("\n----- PPI ANALYSIS -----")
ppi_results, ppi_data = analyze_connectivity_and_save_results(analysis_type='ppi')

print("\nAnalyses complete - results were saved to disk.")