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
    
    try:
        pIPS_map = image.load_img(pIPS_path)
        LO_map = image.load_img(LO_path)
        
        difference_map = image.math_img("img1 - img2", img1=LO_map, img2=pIPS_map)
        return difference_map
    except Exception as e:
        print(f"Error processing {pIPS_path} or {LO_path}: {e}")
        return None

def safe_divide(a, b):
    """
    Perform division safely, handling divide by zero and invalid value warnings.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        c[~np.isfinite(c)] = 0  # replace inf/NaN with 0
    return c

def perform_group_analysis(subs, pIPS_path_template, LO_path_template, output_dir, analysis_type):
    """
    Perform subject-level subtractions and group-level t-tests for both hemispheres.
    Save only the final group t-maps.
    """
    # Dictionary to hold results
    tmap_files = {}
    
    # Process each hemisphere
    for hemi in ['left', 'right']:
        print(f"\nProcessing {hemi} hemisphere for {analysis_type}...")
        
        # Collect subtraction maps in memory
        subtraction_maps = []
        valid_sub_count = 0
        
        # Perform subject-level subtraction
        for subject_id in tqdm(subs, desc=f"Processing subjects"):
            # Construct paths to the individual subject maps
            pIPS_file = pIPS_path_template.format(subject_id=subject_id, hemi=hemi, analysis_type=analysis_type)
            LO_file = LO_path_template.format(subject_id=subject_id, hemi=hemi, analysis_type=analysis_type)
            
            # Perform subtraction
            difference_map = load_and_subtract_fc_maps(pIPS_file, LO_file)
            if difference_map is not None:
                subtraction_maps.append(difference_map)
                valid_sub_count += 1
        
        print(f"Completed processing {valid_sub_count} valid subjects for {hemi} hemisphere")
        
        if not subtraction_maps:
            print(f"Error: No valid subtraction maps for {hemi} hemisphere. Skipping t-test.")
            continue
        
        # Stack all maps into a 4D image
        print(f"Creating 4D image from {len(subtraction_maps)} subtraction maps...")
        stacked_maps = image.concat_imgs(subtraction_maps)
        
        # Perform voxel-wise one-sample t-test
        print("Performing one-sample t-test...")
        data = stacked_maps.get_fdata()
        mean = np.mean(data, axis=-1)
        std = np.std(data, axis=-1)
        n = data.shape[-1]
        t_values = safe_divide(mean, (std / np.sqrt(n)))
        
        # Create and save the t-map
        t_map = image.new_img_like(subtraction_maps[0], t_values)
        t_map_file = os.path.join(output_dir, f'group_ttest_LO_minus_pIPS_{hemi}_{analysis_type}_fdr.nii.gz')
        t_map.to_filename(t_map_file)
        tmap_files[hemi] = t_map_file
        
        print(f"Group-level t-map saved in {t_map_file}")
    
    return tmap_files

def visualize_tmaps(left_tmap_file, right_tmap_file, output_dir, analysis_type):
    """
    Visualize t-maps without additional thresholding since data is already FDR-corrected.
    """
    # Check if files exist
    if not os.path.exists(left_tmap_file) or not os.path.exists(right_tmap_file):
        print("Error: T-map files not found. Cannot create visualizations.")
        return
    
    # Load t-maps
    left_tmap = nib.load(left_tmap_file)
    right_tmap = nib.load(right_tmap_file)
    
    # Set a consistent scale for both hemispheres
    left_data = left_tmap.get_fdata()
    right_data = right_tmap.get_fdata()
    vmax = max(np.max(np.abs(left_data)), np.max(np.abs(right_data)))
    vmin = -vmax
    
    # Create combined figure
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    
    # Plot left hemisphere
    plotting.plot_glass_brain(left_tmap,
                              colorbar=True,
                              plot_abs=False,
                              vmin=vmin, vmax=vmax,
                              title=f'Left Hemisphere: LO minus pIPS ({analysis_type.upper()} t-values)',
                              axes=axes[0])
    
    # Plot right hemisphere
    plotting.plot_glass_brain(right_tmap,
                              colorbar=True,
                              plot_abs=False,
                              vmin=vmin, vmax=vmax,
                              title=f'Right Hemisphere: LO minus pIPS ({analysis_type.upper()} t-values)',
                              axes=axes[1])
    
    # Add overall title
    plt.suptitle(f'LO minus pIPS {analysis_type.upper()} (FDR-corrected)', 
                 fontsize=16, y=0.98)
    
    plt.tight_layout()
    
    # Save combined figure
    combined_fig_path = os.path.join(output_dir, f'LO_minus_pIPS_{analysis_type}_tmap_results_combined_fdr.png')
    fig.savefig(combined_fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Also create individual hemisphere figures
    for hemi, tmap in zip(['left', 'right'], [left_tmap, right_tmap]):
        fig_hemi, ax_hemi = plt.subplots(figsize=(15, 10))
        plotting.plot_glass_brain(tmap,
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
        print(f"  {hemi.capitalize()} hemisphere:")
        print(f"    Max t-value: {np.max(data):.3f}")
        print(f"    Min t-value: {np.min(data):.3f}")
        print(f"    Number of non-zero voxels: {np.sum(data != 0)}")
    
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
    pIPS_path = f"{study_dir}/{{subject_id}}/ses-01/derivatives/fc_mni/{{subject_id}}_pIPS_{{hemi}}_loc_{{analysis_type}}_mni.nii.gz"
    LO_path = f"{study_dir}/{{subject_id}}/ses-01/derivatives/fc_mni/{{subject_id}}_LO_{{hemi}}_loc_{{analysis_type}}_mni.nii.gz"
    
    # Define output directory as requested
    output_dir = f'{curr_dir}/results/group_averages'
    os.makedirs(output_dir, exist_ok=True)
    
    # Define analysis types
    analysis_types = ['fc', 'ppi']
    
    # Process each analysis type
    for analysis_type in analysis_types:
        print(f"\n{'='*80}")
        print(f"Processing {analysis_type.upper()} analysis...")
        print(f"{'='*80}")
        
        # Perform group analysis and get t-map files
        tmap_files = perform_group_analysis(
            subs, 
            pIPS_path, 
            LO_path, 
            output_dir, 
            analysis_type
        )
        
        # Visualize t-maps (without additional thresholding)
        if 'left' in tmap_files and 'right' in tmap_files:
            visualize_tmaps(
                tmap_files['left'],
                tmap_files['right'],
                output_dir,
                analysis_type
            )
        
    print("\nAnalysis and visualization completed for all analysis types.")