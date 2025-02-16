# note to plot surface you need to have freesurfer loaded
import os
import numpy as np
import nibabel as nib
import cortex
import matplotlib.pyplot as plt
import matplotlib as mpl

# Define study directories
study = 'ptoc'
study_dir = f"/lab_data/behrmannlab/vlad/{study}"
results_dir = '/user_data/csimmon2/git_repos/ptoc/results'

def create_flatmap_visualization(roi, hemi, analysis_type='fc', subject='fsaverage'):
    """
    Create flatmap visualization for group-level data
    """
    # Load your group-average MNI-space image
    img_file = f'{results_dir}/group_averages/{roi}_{hemi}_{analysis_type}_avg.nii.gz'
    print(f"Loading: {img_file}")
    
    img = nib.load(img_file)
    data = img.get_fdata()
    print(f"Data shape: {data.shape}")
    
    # Create the volume object with the data
    vol = cortex.Volume(data, subject, 'atlas_2mm', cmap='RdYlBu_r') ### COLOR MAP MUST BE DEFINED HERE TO CHANGE IT
    
    try:
        # Create figure directory if it doesn't exist
        figure_dir = f'{results_dir}/flatmaps'
        os.makedirs(figure_dir, exist_ok=True)

        # Create the flatmap
        fig = cortex.quickflat.make_figure(
            vol,
            with_curvature=True,
            with_rois=True,
            with_labels=True,
            with_colorbar=True,
            threshold=0.12,
            vmin=-np.abs(data).max(),
            vmax=np.abs(data).max(),
            cmap='RdYlBu_r',  # Ensure this is a valid colormap
            size=(1600, 1200)
        )

        # Debug: Check if the colormap is applied
        print(f"Colormap used: {fig.axes[0].images[0].get_cmap().name}")
        
        # Save the figure
        output_file = f'{figure_dir}/{roi}_{hemi}_{analysis_type}_flatmap.png'
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Successfully created flatmap for {roi} {hemi}")
        
    except Exception as e:
        print(f"Error in visualization for {roi} {hemi}: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    # Print diagnostic information
    print(f"Pycortex database: {cortex.database.default_filestore}")
    print(f"Results directory: {results_dir}")
    
    # Define ROIs and hemispheres
    rois = ['pIPS', 'LO']
    hemispheres = ['left', 'right']
    
    # Create flatmaps for each ROI and hemisphere
    for roi in rois:
        for hemi in hemispheres:
            print(f"\nProcessing {roi} {hemi}")
            create_flatmap_visualization(roi, hemi)

if __name__ == "__main__":
    main()