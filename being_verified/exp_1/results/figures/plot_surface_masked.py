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
mask_path = '/user_data/csimmon2/git_repos/ptoc/roiParcels/mruczek_parcels/binary/Left_Hemi_Binary.nii.gz'

def create_flatmap_visualization(roi, hemi, analysis_type='fc', subject='fsaverage'):
    """
    Create flatmap visualization for group-level data with brain mask
    """
    # Load your group-average MNI-space image
    img_file = f'{results_dir}/group_averages/{roi}_{hemi}_{analysis_type}_avg.nii.gz'
    print(f"Loading: {img_file}")
    
    # Load the data
    img = nib.load(img_file)
    data = img.get_fdata()
    print(f"Data shape: {data.shape}")
    
    # Load and check the brain mask
    print(f"Loading mask: {mask_path}")
    mask_img = nib.load(mask_path)
    mask_data = mask_img.get_fdata()
    print(f"Mask shape: {mask_data.shape}")
    
    # Binarize the mask if it's not already binary
    mask_data = (mask_data > 0).astype(float)
    
    # Verify shapes match before masking
    if data.shape != mask_data.shape:
        raise ValueError(f"Data shape {data.shape} doesn't match mask shape {mask_data.shape}")
    
    # Apply the mask
    masked_data = data * mask_data  # Multiply by mask to zero out non-brain regions
    
    # Create the volume object with the masked data
    vol = cortex.Volume(
        masked_data, 
        subject, 
        'atlas_2mm',
        cmap='J5R',
        vmin=-np.abs(masked_data[mask_data > 0]).max(),  
        vmax=np.abs(masked_data[mask_data > 0]).max()
    )
    
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
            size=(160, 120)
        )

        # Debug: Check if the colormap is applied
        print(f"Colormap used: {fig.axes[0].images[0].get_cmap().name}")
        
        # Save the figure
        output_file = f'{figure_dir}/{roi}_{hemi}_{analysis_type}_flatmap_masked.png'
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Successfully created masked flatmap for {roi} {hemi}")
        
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