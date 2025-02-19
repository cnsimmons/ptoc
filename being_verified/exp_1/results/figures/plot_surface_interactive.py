# must run in local
import os
import nibabel as nib
import cortex

# Define directories
results_dir = '/user_data/csimmon2/git_repos/ptoc/results'

def show_interactive_brain(roi, hemi, analysis_type='fc', subject='fsaverage'):
    """
    Open interactive viewer for a specific ROI and hemisphere
    """
    # Construct file path
    img_file = f'{results_dir}/group_averages/{roi}_{hemi}_{analysis_type}_avg.nii.gz'
    print(f"Loading: {img_file}")
    
    # Create the volume object
    vol = cortex.Volume(
        img_file,
        subject,
        'atlas_2mm',
        cmap='YlOrRd_r'  # Red-Yellow colormap
    )
    
    # Open interactive viewer
    cortex.webgl.show(data=vol)

if __name__ == "__main__":
    # Example usage - you can modify these parameters
    show_interactive_brain('pIPS', 'left')