#!/usr/bin/env python3
"""
Partial Correlation Threshold Analysis - Consistent with FC/PPI Pipeline
Applies the same FDR thresholding used in the main FC/PPI analyses
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import image, plotting
from nilearn.glm import threshold_stats_img
import matplotlib.pyplot as plt

# Define directories
study_dir = "/lab_data/behrmannlab/vlad/ptoc"
residual_dir = "/user_data/csimmon2/ptoc_residuals"
results_dir = "/user_data/csimmon2/git_repos/ptoc/results"
curr_dir = "/user_data/csimmon2/git_repos/ptoc"

# FDR parameters (match your established pipeline)
alpha = 0.05
cluster_threshold = 5

def load_subject_maps(subjects, analysis_type, roi, hemisphere):
    """Load subject maps for averaging"""
    all_imgs = []
    valid_subjects = []
    
    for sub in subjects:
        if analysis_type == 'original':
            img_file = f'{study_dir}/{sub}/ses-01/derivatives/fc_mni/{sub}_{roi}_{hemisphere}_loc_fc_mni.nii.gz'
        elif analysis_type == 'cleaned':
            img_file = f'{residual_dir}/{sub}/ses-01/derivatives/fc_mni/{sub}_pIPS_clean_{hemisphere}_loc_fc_mni.nii.gz'
        
        if os.path.exists(img_file):
            all_imgs.append(image.load_img(img_file))
            valid_subjects.append(sub)
        else:
            print(f"Missing: {os.path.basename(img_file)} for {sub}")
    
    return all_imgs, valid_subjects

def apply_consistent_threshold(mean_img, roi, hemisphere, analysis_type):
    """Apply exact thresholding pipeline used in FC/PPI analyses"""
    
    # Z-score the image (like your established pipeline)
    zstat_img = image.math_img("(img-img.mean())/img.std()", img=mean_img)
    
    # Apply FDR threshold (like your established pipeline)
    thresh_result = threshold_stats_img(
        zstat_img, 
        alpha=alpha, 
        height_control='fdr', 
        cluster_threshold=cluster_threshold, 
        two_sided=False
    )
    
    thresh_img, thresh_val = thresh_result[0], thresh_result[1]
    print(f"{roi} {hemisphere} {analysis_type} FDR threshold: {thresh_val:.3f}")
    
    # Zero out negative values (like your established pipeline)
    data = thresh_img.get_fdata()
    data[data <= 0] = 0
    
    # Convert to double precision (like your established pipeline)
    data = data.astype('double')
    
    final_img = nib.Nifti1Image(data, mean_img.affine)
    
    return final_img, thresh_val

def main():
    print("Partial Correlation Analysis with Consistent Thresholding")
    print("=" * 60)
    
    # Load subject info
    sub_info = pd.read_csv(f'{curr_dir}/sub_info.csv')
    subjects = sub_info[sub_info['group'] == 'control']['sub'].tolist()
    print(f"Processing {len(subjects)} control subjects")
    
    # Create output directory
    output_dir = f'{results_dir}/partial_correlation_thresh'
    os.makedirs(output_dir, exist_ok=True)
    
    # Analysis parameters
    hemispheres = ['left', 'right']
    
    # Results storage
    results_list = []
    
    for hemisphere in hemispheres:
        print(f"\nProcessing {hemisphere} hemisphere...")
        
        # Load original pIPS maps
        orig_imgs, orig_subs = load_subject_maps(subjects, 'original', 'pIPS', hemisphere)
        print(f"Found {len(orig_imgs)} original pIPS {hemisphere} maps")
        
        # Load cleaned pIPS maps
        clean_imgs, clean_subs = load_subject_maps(subjects, 'cleaned', 'pIPS', hemisphere)
        print(f"Found {len(clean_imgs)} cleaned pIPS {hemisphere} maps")
        
        # Find common subjects
        common_subs = list(set(orig_subs) & set(clean_subs))
        print(f"Common subjects: {len(common_subs)}")
        
        if len(common_subs) < 5:
            print(f"Warning: Only {len(common_subs)} subjects with both maps for {hemisphere}")
            continue
        
        # Filter to common subjects only
        orig_imgs_common = [orig_imgs[orig_subs.index(sub)] for sub in common_subs]
        clean_imgs_common = [clean_imgs[clean_subs.index(sub)] for sub in common_subs]
        
        # Create mean images
        orig_mean = image.mean_img(orig_imgs_common)
        clean_mean = image.mean_img(clean_imgs_common)
        
        # Apply consistent thresholding
        orig_thresh, orig_thresh_val = apply_consistent_threshold(orig_mean, 'pIPS', hemisphere, 'original')
        clean_thresh, clean_thresh_val = apply_consistent_threshold(clean_mean, 'pIPS_clean', hemisphere, 'cleaned')
        
        # Count significant voxels
        orig_sig_voxels = np.sum(orig_thresh.get_fdata() > 0)
        clean_sig_voxels = np.sum(clean_thresh.get_fdata() > 0)
        
        # Calculate retention
        percent_retained = (clean_sig_voxels / orig_sig_voxels) * 100 if orig_sig_voxels > 0 else 0
        
        print(f"\nResults for {hemisphere} hemisphere:")
        print(f"  Original significant voxels: {orig_sig_voxels}")
        print(f"  Cleaned significant voxels: {clean_sig_voxels}")
        print(f"  Connectivity retention: {percent_retained:.1f}%")
        
        # Save thresholded images
        orig_out = f'{output_dir}/pIPS_{hemisphere}_original_thresh.nii.gz'
        clean_out = f'{output_dir}/pIPS_{hemisphere}_cleaned_thresh.nii.gz'
        
        nib.save(orig_thresh, orig_out)
        nib.save(clean_thresh, clean_out)
        
        print(f"  Saved: {orig_out}")
        print(f"  Saved: {clean_out}")
        
        # Store results
        results_list.append({
            'hemisphere': hemisphere,
            'n_subjects': len(common_subs),
            'orig_threshold': orig_thresh_val,
            'clean_threshold': clean_thresh_val,
            'orig_sig_voxels': orig_sig_voxels,
            'clean_sig_voxels': clean_sig_voxels,
            'percent_retained': percent_retained
        })
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Original
        plotting.plot_glass_brain(orig_thresh, 
                                 title=f'Original pIPS {hemisphere}\n({orig_sig_voxels} voxels, thresh={orig_thresh_val:.3f})',
                                 axes=axes[0],
                                 colorbar=True)
        
        # Cleaned  
        plotting.plot_glass_brain(clean_thresh,
                                 title=f'Cleaned pIPS {hemisphere}\n({clean_sig_voxels} voxels, {percent_retained:.1f}% retained)',
                                 axes=axes[1], 
                                 colorbar=True)
        
        plt.tight_layout()
        fig.suptitle(f'Partial Correlation Analysis - {hemisphere.title()} Hemisphere\n'
                    f'FDR α={alpha}, cluster>{cluster_threshold}', 
                    y=0.98, fontsize=14, fontweight='bold')
        
        plot_out = f'{output_dir}/pIPS_{hemisphere}_comparison.png'
        plt.savefig(plot_out, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {plot_out}")
    
    # Save results summary
    results_df = pd.DataFrame(results_list)
    results_csv = f'{output_dir}/retention_analysis_results.csv'
    results_df.to_csv(results_csv, index=False)
    
    print(f"\n{'='*60}")
    print("SUMMARY RESULTS:")
    print(f"{'='*60}")
    print(results_df.round(3))
    
    # Overall interpretation
    mean_retention = results_df['percent_retained'].mean()
    print(f"\nMean connectivity retention: {mean_retention:.1f}%")
    
    if mean_retention > 70:
        interpretation = "STRONG evidence for dorsal independence"
    elif mean_retention > 30:
        interpretation = "MODERATE evidence for dorsal independence" 
    else:
        interpretation = "LIMITED evidence for dorsal independence"
    
    print(f"Interpretation: {interpretation}")
    print(f"\nResults saved to: {results_csv}")

if __name__ == "__main__":
    main()