# files are zscored in analyses script - here they are thresholded using FDR || use parameter
import os
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import image, plotting
from nilearn.glm import threshold_stats_img
import matplotlib.pyplot as plt

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
                    
                    # Z-score the image before thresholding (like your PI does)
                    zstat_img = image.math_img("(img-img.mean())/img.std()", img=avg_img)
                    
                    # Apply FDR threshold
                    thresh_val = threshold_stats_img(
                        zstat_img, 
                        alpha=alpha, 
                        height_control='fdr', 
                        cluster_threshold=5, 
                        two_sided=False
                    )
                    print(f"{roi} {hemi} {analysis_type} FDR threshold: {thresh_val[1]:.3f}")
                    
                    # Apply threshold and get image
                    thresh_img = image.threshold_img(zstat_img, thresh_val[1])
                    
                    # Zero out negative values
                    data = thresh_img.get_fdata()
                    data[data <= 0] = 0
                    
                    # Convert to double precision (like your PI does)
                    data = data.astype('double')
                    
                    final_img = nib.Nifti1Image(data, avg_img.affine)
                    
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
            fig.text(0.5, 0.01, f'{roi} {analysis_type.upper()} Group Average (FDR α={alpha}, cluster>5)',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
            plt.subplots_adjust(bottom=0.15)
            
            fig.savefig(f'{group_out_dir}/{roi}_{analysis_type}_group_average_thresh.png', 
                        dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved plot for {roi} {analysis_type}")

if __name__ == "__main__":
    main()