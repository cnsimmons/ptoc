# files are zscored in analyses script - here they are thresholded by 2SD and clustered

import os
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import image, plotting
import matplotlib.pyplot as plt
from scipy.ndimage import label

# Define study directories
study = 'ptoc'
study_dir = f"/lab_data/behrmannlab/vlad/{study}"
results_dir = '/user_data/csimmon2/git_repos/ptoc/results'
sub_info_path = '/user_data/csimmon2/git_repos/ptoc/sub_info.csv'

def apply_cluster_threshold(img, min_cluster_size=5):
    data = img.get_fdata()
    binary_map = (data != 0).astype(int)
    labeled_array, num_features = label(binary_map)
    
    for label_id in range(1, num_features + 1):
        cluster_size = np.sum(labeled_array == label_id)
        if cluster_size < min_cluster_size:
            data[labeled_array == label_id] = 0
    
    return nib.Nifti1Image(data, img.affine, img.header)

def main():
    # Define subjects and ROIs
    sub_info = pd.read_csv(sub_info_path)
    subs = sub_info[sub_info['group'] == 'control']['sub'].tolist()
    rois = ['pIPS', 'LO']
    hemispheres = ['left', 'right']
    analysis_types = ['fc', 'ppi']

    # Create output directory
    group_out_dir = f'{results_dir}/group_averages'
    os.makedirs(group_out_dir, exist_ok=True)

    # Process each ROI and analysis type
    for analysis_type in analysis_types:
        # Find global max for consistent visualization within each analysis type
        global_vmax = 0
        
        for roi in rois:
            fig, axes = plt.subplots(1, 2, figsize=(20, 5))
            
            for i, hemi in enumerate(hemispheres):
                print(f"Processing {roi} {hemi} {analysis_type}")
                all_sub_imgs = []
                
                # Collect all subject images
                for sub in subs:
                    img_file = f'{study_dir}/{sub}/ses-01/derivatives/fc_mni/{sub}_{roi}_{hemi}_loc_{analysis_type}_mni.nii.gz'
                    if os.path.exists(img_file):
                        all_sub_imgs.append(image.load_img(img_file))
                
                if all_sub_imgs:
                    # Create average image
                    avg_img = image.mean_img(all_sub_imgs)
                    data = avg_img.get_fdata()
                    
                    # Calculate threshold (mean + 2SD of non-zero values)
                    nonzero_data = data[data != 0]
                    threshold = np.mean(nonzero_data) + 2 * np.std(nonzero_data)
                    print(f"{roi} {hemi} {analysis_type} threshold (mean + 2SD): {threshold:.3f}")
                    
                    # Apply threshold
                    data[data < threshold] = 0
                    thresholded_img = nib.Nifti1Image(data, avg_img.affine)
                    
                    # Apply cluster threshold
                    final_img = apply_cluster_threshold(thresholded_img, min_cluster_size=5)
                    
                    # Update global max for visualization
                    data = final_img.get_fdata()
                    global_vmax = max(global_vmax, np.abs(data).max())
                    
                    # Save thresholded image
                    out_file = f'{group_out_dir}/{roi}_{hemi}_{analysis_type}_thresh.nii.gz'
                    nib.save(final_img, out_file)
                    print(f"Saved thresholded image: {out_file}")

                    # Add to plot
                    display = plotting.plot_glass_brain(
                        final_img,
                        colorbar=True,
                        vmax=global_vmax,
                        vmin=-global_vmax,
                        title=f'{roi} {hemi} {analysis_type.upper()}',
                        axes=axes[i]
                    )
            
            # Save combined hemisphere plot
            plt.tight_layout()
            fig.text(0.5, 0.01, f'{roi} {analysis_type.upper()} Group Average (threshold: mean + 2SD, cluster>5)',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
            plt.subplots_adjust(bottom=0.15)
            
            fig.savefig(f'{group_out_dir}/{roi}_{analysis_type}_group_average_thresh.png', 
                        dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved plot for {roi} {analysis_type}")

if __name__ == "__main__":
    main()