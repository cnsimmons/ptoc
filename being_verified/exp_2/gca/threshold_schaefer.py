#!/usr/bin/env python
# Thresholding script for GCA searchlight results using FDR correction for Experiment 2
# Modified to use Schaefer atlas parcellation

import os
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import image, plotting
from nilearn.glm import threshold_stats_img
from nilearn.maskers import NiftiLabelsMasker
from nilearn.datasets import fetch_atlas_schaefer_2018
import matplotlib.pyplot as plt
import glob
import re

# Define study directories
study = 'ptoc'
study_dir = f"/lab_data/behrmannlab/vlad/{study}"
results_dir = '/user_data/csimmon2/git_repos/ptoc/results'
sub_info_path = '/user_data/csimmon2/git_repos/ptoc/sub_info_tool.csv'

# FDR alpha level
alpha = 0.05

def load_schaefer_atlas(n_rois=200, yeo_networks=17):
    """Load Schaefer atlas with specified number of ROIs and networks"""
    atlas = fetch_atlas_schaefer_2018(
        n_rois=n_rois, 
        yeo_networks=yeo_networks,
        resolution_mm=2
    )
    return atlas['maps'], atlas['labels']

def parcellate_data(img, atlas_img, masker):
    """Parcellate an image using the provided masker"""
    # Resample atlas to match image dimensions if needed
    resampled_atlas = image.resample_to_img(atlas_img, img)
    
    # Parcellate the data
    parcel_data = masker.fit_transform(img)
    return parcel_data.squeeze()

def create_parcellated_image(parcel_data, atlas_img, masker):
    """Create a Nifti image from parcellated data"""
    # Inverse transform the parcellated data back to image space
    img_data = masker.inverse_transform(parcel_data.reshape(1, -1))
    return img_data

def main():
    # Load Schaefer atlas
    atlas_img, atlas_labels = load_schaefer_atlas()
    masker = NiftiLabelsMasker(labels_img=atlas_img, standardize=False)
    
    # Define subjects and ROIs
    sub_info = pd.read_csv(sub_info_path)
    subs = sub_info[sub_info['exp'] == 'spaceloc']['sub'].tolist()
    
    # Define ROIs, hemispheres, and conditions
    rois = ['pIPS', 'LO']
    hemispheres = ['left', 'right']
    conditions = ['tool', 'nontool']
    
    # Create output directory
    group_out_dir = f'{results_dir}/gca_group_averages_exp2_schaefer'
    os.makedirs(group_out_dir, exist_ok=True)
    
    # Create dictionaries to store images for visualization
    all_thresholded_imgs = {}
    
    # Process each condition
    for condition in conditions:
        # Find global max for each condition for consistent visualization
        global_max = 0
        
        # Process each ROI-hemisphere combination
        for roi in rois:
            for hemi in hemispheres:
                roi_hemi = f"{roi}_{hemi}"
                condition_key = f"{condition}_{roi_hemi}"
                print(f"Processing {condition} {roi_hemi}")
                
                # Collect all subject images
                all_sub_imgs = []
                for sub in subs:
                    # Looking for combined object GCA files
                    img_file = f'{study_dir}/{sub}/ses-01/derivatives/gca/combined_{condition}_{roi_hemi}_mni_1217.nii.gz'
                    if os.path.exists(img_file):
                        try:
                            img = image.load_img(img_file)
                            # Check if image is in MNI space (91x109x91)
                            if img.shape == (91, 109, 91):  # Standard MNI shape
                                # Parcellate the image
                                parcel_data = parcellate_data(img, atlas_img, masker)
                                # Create image from parcellated data
                                parcel_img = create_parcellated_image(parcel_data, atlas_img, masker)
                                all_sub_imgs.append(parcel_img)
                            else:
                                print(f"Warning: {img_file} doesn't appear to be in MNI space (shape: {img.shape})")
                        except Exception as e:
                            print(f"Error loading {img_file}: {e}")
                    else:
                        print(f"Missing file for {sub}, {condition} {roi_hemi}")
                
                if not all_sub_imgs:
                    print(f"No images found for {condition} {roi_hemi}, skipping")
                    continue
                
                # Smooth the parcellated images
                smooth_sub_imgs = [image.smooth_img(img, fwhm=8) for img in all_sub_imgs]

                # Create average image from SMOOTHED images
                print(f"Creating average from {len(smooth_sub_imgs)} subjects")
                avg_img = image.mean_img(smooth_sub_imgs)

                # Z-score the image before thresholding
                zstat_img = image.math_img("(img-img.mean())/img.std()", img=avg_img)

                # Apply FDR threshold
                thresh_val = threshold_stats_img(
                    zstat_img, 
                    alpha=alpha, 
                    height_control='fdr', 
                    cluster_threshold=15, 
                    two_sided=False
                )
                print(f"{condition} {roi_hemi} FDR threshold: {thresh_val[1]:.3f}")
                
                # Apply threshold and get image
                thresh_img = image.threshold_img(zstat_img, thresh_val[1])
                
                # Convert to double precision
                data = thresh_img.get_fdata()
                data = data.astype('double')
                
                final_img = nib.Nifti1Image(data, avg_img.affine)
                
                # Save thresholded image
                out_file = f'{group_out_dir}/{condition}_{roi_hemi}_gca_thresh_schaefer.nii.gz'
                nib.save(final_img, out_file)
                print(f"Saved thresholded image: {out_file}")
                
                # Store for the combined visualization
                all_thresholded_imgs[condition_key] = final_img
                
                # Update global max for visualization
                data_max = np.abs(data).max()
                if data_max > global_max:
                    global_max = data_max
        
        # Create visualizations for each condition
        if any(key.startswith(condition) for key in all_thresholded_imgs):
            create_2x2_visualization(
                condition, 
                all_thresholded_imgs, 
                global_max, 
                group_out_dir, 
                rois, 
                hemispheres
            )
        else:
            print(f"No thresholded images to visualize for {condition}")

def create_2x2_visualization(condition, thresholded_imgs, global_max, output_dir, rois, hemispheres):
    """Create a 2x2 visualization of thresholded GCA images for a specific condition"""
    
    # Create figure
    fig = plt.figure(figsize=(15, 7))
    
    # Add column labels (hemisphere labels)
    fig.text(0.25, 0.89, 'Left Hemisphere', ha='center', va='bottom', fontsize=14)
    fig.text(0.75, 0.89, 'Right Hemisphere', ha='center', va='bottom', fontsize=14)
    
    # Add row labels
    fig.text(0.07, 0.75, 'pIPS', ha='right', va='center', fontsize=14)
    fig.text(0.07, 0.25, 'LO', ha='right', va='center', fontsize=14)
    
    # Define the order for 2x2 layout
    positions = {
        'pIPS_left': 1,
        'pIPS_right': 2,
        'LO_left': 3,
        'LO_right': 4
    }
    
    # Create a list to store display objects
    displays = []
    
    for roi in rois:
        for hemi in hemispheres:
            roi_hemi = f"{roi}_{hemi}"
            condition_key = f"{condition}_{roi_hemi}"
            
            # Calculate subplot position
            position = positions[roi_hemi]
            ax = plt.subplot(2, 2, position)
            
            if condition_key in thresholded_imgs:
                # Display the brain with specific display settings
                display = plotting.plot_glass_brain(
                    thresholded_imgs[condition_key],
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
    fig.suptitle(f'Group Average GCA Results - {condition.capitalize()} Condition\nFDR Corrected (α=0.05, cluster>15) - Schaefer Atlas', 
                y=.99, fontsize=14)
    
    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.4, hspace=0.3, top=0.9, bottom=0.1, left=0.1, right=0.9)
    
    # Save the figure
    plt.savefig(f'{output_dir}/gca_{condition}_combined_brain_visualizations_schaefer.png', dpi=300, bbox_inches='tight')
    print(f"Saved {condition} visualization to {output_dir}/gca_{condition}_combined_brain_visualizations_schaefer.png")
    plt.close(fig)

if __name__ == "__main__":
    main()