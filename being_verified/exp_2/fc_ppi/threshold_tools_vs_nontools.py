# tools_vs_nontools_threshold.py
import os
import sys
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import image, plotting
from nilearn.glm import threshold_stats_img
import matplotlib.pyplot as plt
import logging

# Define study directories to match your other scripts
curr_dir = '/user_data/csimmon2/git_repos/ptoc'
study_dir = "/lab_data/behrmannlab/vlad/ptoc"
results_dir = "/user_data/csimmon2/git_repos/ptoc/results/tools"
sub_info_path = f'{curr_dir}/sub_info_tool.csv'

# FDR alpha level
alpha = 0.05

def setup_logging():
    """Configure logging for the script"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def main():
    logger = setup_logging()
    logger.info("Starting tools vs nontools PPI thresholding")
    
    # Define subjects and ROIs
    sub_info = pd.read_csv(sub_info_path)
    subs = sub_info[sub_info['exp'] == 'spaceloc']['sub'].tolist()
    rois = ['pIPS', 'LO']
    hemispheres = ['left', 'right']
    
    # Create output directory
    group_out_dir = f'{results_dir}/group_tools_vs_nontools'
    os.makedirs(group_out_dir, exist_ok=True)
    
    for roi in rois:
        # Create a single figure for both hemispheres
        fig, axes = plt.subplots(1, 2, figsize=(20, 5))
        
        for i, hemi in enumerate(hemispheres):
            logger.info(f"===== Processing {roi} {hemi} tools vs nontools PPI =====")
            all_sub_imgs = []
            
            # File pattern matching the output of your MNI conversion script
            file_pattern = f"{study_dir}/{{sub}}/ses-01/derivatives/ppi/mni/{{sub}}_{roi}_{hemi}_tools_vs_nontools_ToolLoc_ppi_mni.nii.gz"
            
            logger.info(f"Looking for files with pattern: {file_pattern.format(sub='sub-XXX')}")
            
            found_files = 0
            missing_files = 0
            invalid_files = 0
            
            # Collect all subject images
            for sub in subs:
                img_file = file_pattern.format(sub=sub)
                
                if os.path.exists(img_file):
                    found_files += 1
                    logger.info(f"Found file: {img_file}")
                    try:
                        # Load and check image dimensions
                        img = image.load_img(img_file)
                        
                        # Check if image is in MNI space (91x109x91)
                        if img.shape == (91, 109, 91):  # Standard MNI shape
                            all_sub_imgs.append(img)
                        else:
                            invalid_files += 1
                            logger.warning(f"Warning: {img_file} doesn't appear to be in MNI space (shape: {img.shape})")
                    except Exception as e:
                        invalid_files += 1
                        logger.error(f"Error loading {img_file}: {e}")
                else:
                    missing_files += 1
                    # Only print first few missing files to avoid console clutter
                    if missing_files <= 5:
                        logger.warning(f"Missing file: {img_file}")
                    elif missing_files == 6:
                        logger.warning("Additional missing files not shown...")
            
            logger.info(f"Summary: Found: {found_files}, Missing: {missing_files}, Invalid: {invalid_files} files")
            
            if all_sub_imgs:
                # Create average image
                logger.info(f"Creating average from {len(all_sub_imgs)} subjects")
                avg_img = image.mean_img(all_sub_imgs)
                
                # Z-score the image before thresholding
                zstat_img = image.math_img("(img-img.mean())/img.std()", img=avg_img)
                
                # Apply two-sided FDR threshold
                thresh_val = threshold_stats_img(
                    zstat_img, 
                    alpha=alpha, 
                    height_control='fdr', 
                    cluster_threshold=5, 
                    two_sided=True  # Two-sided for both positive and negative
                )
                logger.info(f"{roi} {hemi} tools vs nontools PPI FDR threshold: {thresh_val[1]:.3f}")
                
                # Save the raw z-statistic image (before thresholding)
                raw_out_file = f'{group_out_dir}/{roi}_{hemi}_tools_vs_nontools_ppi_zstat.nii.gz'
                nib.save(zstat_img, raw_out_file)
                logger.info(f"Saved raw z-statistic image: {raw_out_file}")
                
                # Manual thresholding for both positive and negative values
                data = zstat_img.get_fdata()
                threshold = thresh_val[1]
                
                # Create a mask for values that don't pass the threshold (both positive and negative)
                mask = np.abs(data) < threshold
                
                # Apply the mask to create thresholded data
                thresholded_data = data.copy()
                thresholded_data[mask] = 0
                
                # Create and save thresholded image
                thresholded_img = nib.Nifti1Image(thresholded_data.astype('double'), zstat_img.affine)
                combined_out_file = f'{group_out_dir}/{roi}_{hemi}_tools_vs_nontools_ppi_thresh.nii.gz'
                nib.save(thresholded_img, combined_out_file)
                logger.info(f"Saved thresholded image: {combined_out_file}")
                
                # Use glass brain for visualization
                # We'll manually create a positive and negative version just for display
                pos_data = thresholded_data.copy()
                pos_data[pos_data < 0] = 0
                pos_img = nib.Nifti1Image(pos_data, zstat_img.affine)
                
                neg_data = thresholded_data.copy()
                neg_data[neg_data > 0] = 0
                neg_data = -neg_data  # Make negative values positive for visualization
                neg_img = nib.Nifti1Image(neg_data, zstat_img.affine)
                
                # Plot positive values
                display_pos = plotting.plot_glass_brain(
                    pos_img,
                    colorbar=True,
                    display_mode='ortho',
                    axes=axes[i],
                    title=f'{roi} {hemi}',
                    cmap='hot'  # Red-yellow for positive values
                )
                
                # Overlay negative values on the same plot
                display_pos.add_overlay(neg_img, cmap='winter')  # Blue for negative values
                
            else:
                logger.warning(f"No valid images found for {roi} {hemi} tools vs nontools PPI")
        
        # Save the combined figure
        plt.tight_layout()
        fig.text(0.5, 0.01, f'{roi} Tools vs Nontools PPI (FDR α={alpha}, cluster>5, Red=Tools>Nontools, Blue=Nontools>Tools)',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
        plt.subplots_adjust(bottom=0.15)
        fig.savefig(f'{group_out_dir}/{roi}_tools_vs_nontools_ppi_thresh.png', 
                    dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved combined plot for {roi} tools vs nontools PPI")

if __name__ == "__main__":
    main()