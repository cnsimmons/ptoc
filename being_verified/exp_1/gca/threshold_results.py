#!/usr/bin/env python
# Thresholding script for GCA searchlight results using FDR correction

import os
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import image, plotting
from nilearn.glm import threshold_stats_img
import matplotlib.pyplot as plt
import glob
import re

# Define study directories
study = 'ptoc'
study_dir = f"/lab_data/behrmannlab/vlad/{study}"
results_dir = '/user_data/csimmon2/git_repos/ptoc/results'
sub_info_path = '/user_data/csimmon2/git_repos/ptoc/sub_info.csv'

# FDR alpha level
alpha = 0.05

def main():
    # Define subjects and ROIs
    sub_info = pd.read_csv(sub_info_path)
    subs = sub_info[sub_info['group'] == 'control']['sub'].tolist()
    
    # Define ROIs and hemispheres in the format they appear in your filenames
    rois_hemispheres = ['pIPS_left', 'pIPS_right', 'LO_left', 'LO_right']
    
    # Create output directory
    group_out_dir = f'{results_dir}/gca_group_averages'
    os.makedirs(group_out_dir, exist_ok=True)
    
    # Process each ROI-hemisphere combination
    all_thresholded_imgs = {}
    global_max = 0
    
    for roi_hemi in rois_hemispheres:
        print(f"Processing {roi_hemi}")
        
        # Collect all subject images
        all_sub_imgs = []
        for sub in subs:
            # Looking for combined object GCA files as in your plotting script
            img_file = f'{study_dir}/{sub}/ses-01/derivatives/gca/combined_object_{roi_hemi}_mni.nii.gz'
            if os.path.exists(img_file):
                all_sub_imgs.append(image.load_img(img_file))
            else:
                print(f"Missing file for {sub}, {roi_hemi}")
        
        if not all_sub_imgs:
            print(f"No images found for {roi_hemi}, skipping")
            continue
        
        # Create average image
        avg_img = image.mean_img(all_sub_imgs)
        
        # Z-score the image before thresholding (like in your PPI/FC script)
        zstat_img = image.math_img("(img-img.mean())/img.std()", img=avg_img)
        
        # Apply FDR threshold
        thresh_val = threshold_stats_img(
            zstat_img, 
            alpha=alpha, 
            height_control='fdr', 
            cluster_threshold=5, 
            two_sided=False
        )
        print(f"{roi_hemi} FDR threshold: {thresh_val[1]:.3f}")
        
        # Apply threshold and get image
        thresh_img = image.threshold_img(zstat_img, thresh_val[1])
        
        # Convert to double precision (like in your PPI/FC script)
        data = thresh_img.get_fdata()
        data = data.astype('double')
        
        final_img = nib.Nifti1Image(data, avg_img.affine)
        
        # Save thresholded image
        out_file = f'{group_out_dir}/{roi_hemi}_gca_thresh.nii.gz'
        nib.save(final_img, out_file)
        print(f"Saved thresholded image: {out_file}")
        
        # Store for the combined visualization
        all_thresholded_imgs[roi_hemi] = final_img
        
        # Update global max for visualization
        data_max = np.abs(data).max()
        if data_max > global_max:
            global_max = data_max

    # Create a 2x2 combined visualization like in your plotting script
    if all_thresholded_imgs:
        create_2x2_visualization(all_thresholded_imgs, global_max, group_out_dir)
    else:
        print("No thresholded images to visualize")

def create_2x2_visualization(thresholded_imgs, global_max, output_dir):
    """Create a 2x2 visualization of thresholded GCA images"""
    
    # Create figure
    fig = plt.figure(figsize=(15, 7))
    
    # Add column labels (hemisphere labels)
    fig.text(0.25, 0.89, 'Left Hemisphere', ha='center', va='bottom', fontsize=14)
    fig.text(0.75, 0.89, 'Right Hemisphere', ha='center', va='bottom', fontsize=14)
    
    # Add row labels
    fig.text(0.07, 0.75, 'pIPS', ha='right', va='center', fontsize=14)
    fig.text(0.07, 0.25, 'LO', ha='right', va='center', fontsize=14)
    
    # Define the order for 2x2 layout
    region_order = ['pIPS_left', 'pIPS_right', 'LO_left', 'LO_right']
    
    # Create a list to store display objects
    displays = []
    
    for idx, region in enumerate(region_order):
        # Calculate subplot position (2x2 grid)
        ax = plt.subplot(2, 2, idx + 1)
        
        if region in thresholded_imgs:
            # Display the brain with specific display settings
            display = plotting.plot_glass_brain(
                thresholded_imgs[region],
                colorbar=True,
                plot_abs=False,
                vmin=-global_max,
                vmax=global_max,
                display_mode='ortho',
                axes=ax,
                black_bg=False,
                title=''  # Remove individual titles since we have row/column labels
            )
            displays.append(display)
    
    # Add colorbar label to the last display
    if displays:
        # Get the last colorbar axes
        last_display = displays[-1]
        if hasattr(last_display, '_cbar'):
            cbar = last_display._cbar
            # Update colorbar label
            cbar.set_label('Z-score (FDR corrected)', rotation=270, labelpad=25, fontsize=12)
    
    # Add a title for the entire figure
    fig.suptitle('Group Average GCA Results\nFDR Corrected (α=0.05, cluster>5)', 
                y=.99, fontsize=14)
    
    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.4, hspace=0.3, top=0.9, bottom=0.1, left=0.1, right=0.9)
    
    # Save the figure
    plt.savefig(f'{output_dir}/gca_combined_brain_visualizations.png', dpi=300, bbox_inches='tight')
    print(f"Saved combined visualization to {output_dir}/gca_combined_brain_visualizations.png")

if __name__ == "__main__":
    main()