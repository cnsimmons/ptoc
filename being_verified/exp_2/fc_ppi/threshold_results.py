# Updated threshold_results.py for experiment 2
import os
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import image, plotting
from nilearn.glm import threshold_stats_img
import matplotlib.pyplot as plt
import glob  # Added for more flexible file searching

# Define study directories
study_dir = "/lab_data/behrmannlab/vlad/ptoc"
results_dir = '/user_data/csimmon2/git_repos/ptoc/results'
sub_info_path = '/user_data/csimmon2/git_repos/ptoc/sub_info_tool.csv'
output_dir = '/user_data/csimmon2/git_repos/ptoc/tools'

# FDR alpha level
alpha = 0.05

def main():
    # Define subjects and ROIs
    sub_info = pd.read_csv(sub_info_path)
    subs = sub_info[sub_info['exp'] == 'spaceloc']['sub'].tolist()
    rois = ['pIPS', 'LO']
    hemispheres = ['left', 'right']
    conditions = ['tools', 'nontools']
    analysis_types = ['fc', 'ppi']

    # Create output directory
    group_out_dir = f'{results_dir}/group_averages_exp2'
    os.makedirs(group_out_dir, exist_ok=True)
    
    # Process each condition, analysis type, roi, and hemisphere
    for condition in conditions:
        for analysis_type in analysis_types:
            # Find global max for consistent visualization
            global_vmax = 0
            
            for roi in rois:
                fig, axes = plt.subplots(1, 2, figsize=(20, 5))
                
                for i, hemi in enumerate(hemispheres):
                    print(f"\n\n===== Processing {roi} {hemi} {condition} {analysis_type} =====")
                    all_sub_imgs = []
                    
                    # Define pattern based on condition
                    if condition == 'tools':
                        # For tools, use the standard pattern (no condition in name)
                        file_pattern = f'{study_dir}/{{sub}}/ses-01/derivatives/{analysis_type}/mni/{{sub}}_{roi}_{hemi}_ToolLoc_{analysis_type}_mni.nii.gz'
                    else:  # nontools
                        # For nontools, include condition in file name
                        file_pattern = f'{study_dir}/{{sub}}/ses-01/derivatives/{analysis_type}/mni/{{sub}}_{roi}_{hemi}_{condition}_ToolLoc_{analysis_type}_mni.nii.gz'
                    
                    # Flexible path alternative (checks for file in multiple locations)
                    def find_file(sub, pattern):
                        # Primary pattern
                        file_path = pattern.format(sub=sub)
                        if os.path.exists(file_path):
                            return file_path
                            
                        # Alternative locations to check
                        alternatives = [
                            f'{study_dir}/{sub}/ses-01/derivatives/{analysis_type}/mni/{sub}_{roi}_{hemi}{"_"+condition if condition=="nontools" else ""}_ToolLoc_{analysis_type}_mni.nii.gz',
                            f'{study_dir}/{sub}/derivatives/{analysis_type}/mni/{sub}_{roi}_{hemi}{"_"+condition if condition=="nontools" else ""}_ToolLoc_{analysis_type}_mni.nii.gz',
                            f'{study_dir}/{sub}/ses-01/derivatives/{analysis_type}/{sub}_{roi}_{hemi}{"_"+condition if condition=="nontools" else ""}_ToolLoc_{analysis_type}_mni.nii.gz'
                        ]
                        
                        for alt in alternatives:
                            if os.path.exists(alt):
                                print(f"Found file at alternative location: {alt}")
                                return alt
                                
                        # If nothing found, return the original path (which doesn't exist)
                        return file_path
                    
                    print(f"Looking for files with pattern: {file_pattern.format(sub='sub-XXX')}")
                    
                    found_files = 0
                    missing_files = 0
                    invalid_files = 0
                    
                    # Collect all subject images
                    for sub in subs:
                        img_file = find_file(sub, file_pattern)
                        
                        if os.path.exists(img_file):
                            found_files += 1
                            print(f"Found file: {img_file}")
                            try:
                                # Load and check image dimensions
                                img = image.load_img(img_file)
                                
                                # Check if image is in MNI space (91x109x91)
                                if img.shape == (91, 109, 91):  # Standard MNI shape
                                    all_sub_imgs.append(img)
                                else:
                                    invalid_files += 1
                                    print(f"Warning: {img_file} doesn't appear to be in MNI space (shape: {img.shape})")
                            except Exception as e:
                                invalid_files += 1
                                print(f"Error loading {img_file}: {e}")
                        else:
                            missing_files += 1
                            # Only print first few missing files to avoid console clutter
                            if missing_files <= 5:
                                print(f"Missing file: {img_file}")
                            elif missing_files == 6:
                                print("Additional missing files not shown...")
                    
                    print(f"Summary: Found: {found_files}, Missing: {missing_files}, Invalid: {invalid_files} files")
                    
                    if all_sub_imgs:
                        # Create average image
                        print(f"Creating average from {len(all_sub_imgs)} subjects")
                        avg_img = image.mean_img(all_sub_imgs)
                        
                        # Z-score the image before thresholding
                        zstat_img = image.math_img("(img-img.mean())/img.std()", img=avg_img)
                        
                        # Apply FDR threshold
                        thresh_val = threshold_stats_img(
                            zstat_img, 
                            alpha=alpha, 
                            height_control='fdr', 
                            cluster_threshold=5, 
                            two_sided=False
                        )
                        print(f"{roi} {hemi} {condition} {analysis_type} FDR threshold: {thresh_val[1]:.3f}")
                        
                        # Apply threshold and get image
                        thresh_img = image.threshold_img(zstat_img, thresh_val[1])
                        
                        # Zero out negative values
                        data = thresh_img.get_fdata()
                        data[data <= 0] = 0
                        
                        # Convert to double precision
                        data = data.astype('double')
                        
                        final_img = nib.Nifti1Image(data, avg_img.affine)
                        
                        # Update global max for visualization
                        data = final_img.get_fdata()
                        current_max = np.abs(data).max()
                        if current_max > global_vmax:
                            global_vmax = current_max
                        
                        # Save thresholded image
                        out_file = f'{group_out_dir}/{roi}_{hemi}_{condition}_{analysis_type}_thresh.nii.gz'
                        nib.save(final_img, out_file)
                        print(f"Saved thresholded image: {out_file}")

                        # Add to plot
                        display = plotting.plot_glass_brain(
                            final_img,
                            colorbar=True,
                            title=f'{roi} {hemi} {condition} {analysis_type.upper()}',
                            axes=axes[i]
                        )
                    else:
                        print(f"No valid images found for {roi} {hemi} {condition} {analysis_type}")
                
                # Set vmax for consistent visualization
                for ax in axes:
                    if hasattr(ax, 'figure_') and hasattr(ax.figure_, 'cbar'):
                        ax.figure_.cbar.set_clim(0, global_vmax)
                
                # Save combined hemisphere plot
                plt.tight_layout()
                fig.text(0.5, 0.01, f'{roi} {condition} {analysis_type.upper()} Group Average (FDR α={alpha}, cluster>5)',
                        ha='center', va='bottom', fontsize=12, fontweight='bold')
                plt.subplots_adjust(bottom=0.15)
                
                fig.savefig(f'{group_out_dir}/{roi}_{condition}_{analysis_type}_group_average_thresh.png', 
                            dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f"Saved plot for {roi} {condition} {analysis_type}")

if __name__ == "__main__":
    main()