import sys
curr_dir = '/user_data/csimmon2/git_repos/ptoc'
sys.path.insert(0, curr_dir)

import numpy as np
from nilearn import image
from scipy import stats
import pandas as pd
import os
from tqdm import tqdm
import warnings
import nibabel as nib
from nilearn import plotting
import matplotlib.pyplot as plt

def load_and_subtract_fc_maps(pIPS_path, LO_path):
    """
    Load pIPS and LO functional connectivity maps and subtract them.
    """
    # Check if files exist
    if not os.path.exists(pIPS_path) or not os.path.exists(LO_path):
        return None
    
    pIPS_map = image.load_img(pIPS_path)
    LO_map = image.load_img(LO_path)
    
    difference_map = image.math_img("img1 - img2", img1=LO_map, img2=pIPS_map)
    return difference_map

def safe_divide(a, b):
    """
    Perform division safely, handling divide by zero and invalid value warnings.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        c[~np.isfinite(c)] = 0  # replace inf/NaN with 0
    return c

def subject_level_analysis(subject_id, pIPS_path, LO_path, output_dir, hemi, analysis_type):
    """
    Perform subject-level subtraction using FDR-corrected maps and save the result.
    """
    # Construct paths to the FDR-corrected maps for this subject
    pIPS_file = pIPS_path.format(subject_id=subject_id, hemi=hemi, analysis_type=analysis_type)
    LO_file = LO_path.format(subject_id=subject_id, hemi=hemi, analysis_type=analysis_type)
    
    difference_map = load_and_subtract_fc_maps(pIPS_file, LO_file)
    if difference_map is None:
        print(f"Warning: Files not found for subject {subject_id}, hemisphere {hemi}. Skipping.")
        return None

    # Save the subtraction result in the output directory
    subtraction_file = os.path.join(output_dir, f'{subject_id}_LO_minus_pIPS_{hemi}_{analysis_type}_fdr.nii.gz')
    difference_map.to_filename(subtraction_file)
    
    print(f"Subtraction complete for subject {subject_id}, hemisphere {hemi}, {analysis_type}. Result saved in {subtraction_file}")
    return subtraction_file

def group_level_analysis(subtraction_files, output_dir, hemi, analysis_type):
    """
    Perform group-level t-test on subject subtraction maps and save the result.
    """
    if not subtraction_files:
        print(f"Error: No valid subtraction files for {hemi} hemisphere. Skipping t-test.")
        return None

    # Load all subtraction maps
    all_maps = [image.load_img(f) for f in subtraction_files]
    
    # Stack all maps into a 4D image
    stacked_maps = image.concat_imgs(all_maps)

    # Perform voxel-wise one-sample t-test
    data = stacked_maps.get_fdata()
    mean = np.mean(data, axis=-1)
    std = np.std(data, axis=-1)
    n = data.shape[-1]
    t_values = safe_divide(mean, (std / np.sqrt(n)))

    # Create and save the t-map
    t_map = image.new_img_like(stacked_maps, t_values)
    t_map_file = os.path.join(output_dir, f'group_ttest_LO_minus_pIPS_{hemi}_{analysis_type}_fdr.nii.gz')
    t_map.to_filename(t_map_file)

    print(f"Group-level analysis complete for {hemi} hemisphere, {analysis_type}. T-map saved in {t_map_file}")
    return t_map_file

def visualize_tmaps(left_tmap_file, right_tmap_file, output_dir, analysis_type, threshold=3.5):
    """
    Visualize t-maps with proper thresholding and colorbars.
    """
    # Check if files exist
    if not os.path.exists(left_tmap_file) or not os.path.exists(right_tmap_file):
        print("Error: T-map files not found. Cannot create visualizations.")
        return

    # Function to load and threshold t-map
    def load_and_threshold_tmap(file_path, threshold):
        img = nib.load(file_path)
        data = img.get_fdata()
        thresholded_data = np.where(np.abs(data) > threshold, data, 0)
        return nib.Nifti1Image(thresholded_data, img.affine, img.header)

    # Load and threshold t-maps
    left_tmap_thresholded = load_and_threshold_tmap(left_tmap_file, threshold)
    right_tmap_thresholded = load_and_threshold_tmap(right_tmap_file, threshold)

    # Set a consistent scale for both hemispheres
    vmax = max(np.max(np.abs(left_tmap_thresholded.get_fdata())), 
               np.max(np.abs(right_tmap_thresholded.get_fdata())))
    vmin = -vmax

    # Create combined figure
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    
    # Plot left hemisphere
    plotting.plot_glass_brain(left_tmap_thresholded,
                              threshold=threshold,
                              colorbar=True,
                              plot_abs=False,
                              vmin=vmin, vmax=vmax,
                              title=f'Left Hemisphere: LO minus pIPS ({analysis_type.upper()} t-values)',
                              axes=axes[0])
    
    # Plot right hemisphere
    plotting.plot_glass_brain(right_tmap_thresholded,
                              threshold=threshold,
                              colorbar=True,
                              plot_abs=False,
                              vmin=vmin, vmax=vmax,
                              title=f'Right Hemisphere: LO minus pIPS ({analysis_type.upper()} t-values)',
                              axes=axes[1])
    
    # Add overall title
    plt.suptitle(f'LO minus pIPS {analysis_type.upper()} (FDR-corrected, threshold t > {threshold})', 
                 fontsize=16, y=0.98)
    
    plt.tight_layout()
    
    # Save combined figure
    combined_fig_path = os.path.join(output_dir, f'LO_minus_pIPS_{analysis_type}_tmap_results_combined_fdr.png')
    fig.savefig(combined_fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Also create individual hemisphere figures
    for hemi, tmap in zip(['left', 'right'], [left_tmap_thresholded, right_tmap_thresholded]):
        fig_hemi, ax_hemi = plt.subplots(figsize=(15, 10))
        plotting.plot_glass_brain(tmap,
                                  threshold=threshold,
                                  colorbar=True,
                                  plot_abs=False,
                                  vmin=vmin, vmax=vmax,
                                  title=f'{hemi.capitalize()} Hemisphere: LO minus pIPS ({analysis_type.upper()} t-values, FDR-corrected)',
                                  axes=ax_hemi)
        plt.tight_layout()
        
        # Save the hemisphere figure
        hemi_fig_path = os.path.join(output_dir, f'LO_minus_pIPS_{analysis_type}_tmap_results_{hemi}_fdr.png')
        fig_hemi.savefig(hemi_fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig_hemi)
    
    # Print statistics
    print(f"\nVisualization statistics for {analysis_type}:")
    for hemi, tmap_file in zip(['left', 'right'], [left_tmap_file, right_tmap_file]):
        tmap = nib.load(tmap_file)
        data = tmap.get_fdata()
        significant_voxels = np.sum(np.abs(data) > threshold)
        print(f"  {hemi.capitalize()} hemisphere:")
        print(f"    Max t-value: {np.max(data):.3f}")
        print(f"    Min t-value: {np.min(data):.3f}")
        print(f"    Number of significant voxels: {significant_voxels}")
    
    print(f"\n  Scale range: {vmin:.3f} to {vmax:.3f}")
    print(f"  Combined figure saved as: {combined_fig_path}")

if __name__ == "__main__":
    # Define parameters
    study = 'ptoc'
    study_dir = f"/lab_data/behrmannlab/vlad/{study}"
    
    # Load subject info
    sub_info = pd.read_csv(f'{curr_dir}/sub_info.csv')
    subs = sub_info[sub_info['group'] == 'control']['sub'].tolist()
    
    # Define paths for input files (using FDR-corrected maps from the first script)
    # Update these paths to point to the location of your FDR-corrected individual subject maps
    pIPS_path = f"{study_dir}/{{subject_id}}/ses-01/derivatives/fc_mni/{{subject_id}}_pIPS_{{hemi}}_loc_{{analysis_type}}_mni.nii.gz"
    LO_path = f"{study_dir}/{{subject_id}}/ses-01/derivatives/fc_mni/{{subject_id}}_LO_{{hemi}}_loc_{{analysis_type}}_mni.nii.gz"
    
    # Define output directories
    subtraction_out_dir = f'{curr_dir}/analyses/subtraction_fdr_corrected'
    os.makedirs(subtraction_out_dir, exist_ok=True)
    
    # Define analysis types
    analysis_types = ['fc', 'ppi']
    
    # Process each analysis type
    for analysis_type in analysis_types:
        print(f"\nProcessing {analysis_type.upper()} analysis...")
        
        # Perform subject-level analysis and collect subtraction files
        subtraction_files = {'left': [], 'right': []}
        for subject_id in tqdm(subs, desc=f"Processing subjects for {analysis_type}"):
            for hemi in ['left', 'right']:
                subtraction_file = subject_level_analysis(
                    subject_id, 
                    pIPS_path, 
                    LO_path, 
                    subtraction_out_dir, 
                    hemi,
                    analysis_type
                )
                if subtraction_file:
                    subtraction_files[hemi].append(subtraction_file)
        
        # Perform group-level analysis
        tmap_files = {}
        for hemi in ['left', 'right']:
            tmap_files[hemi] = group_level_analysis(
                subtraction_files[hemi], 
                subtraction_out_dir, 
                hemi,
                analysis_type
            )
        
        # Visualize t-maps
        if tmap_files['left'] and tmap_files['right']:
            visualize_tmaps(
                tmap_files['left'],
                tmap_files['right'],
                subtraction_out_dir,
                analysis_type,
                threshold=3.5  # Adjust this threshold as needed
            )
        
    print("\nAnalysis and visualization completed for all analysis types.")