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

def create_flatmap_visualization(roi, hemi, analysis_type='fc', subject='fsaverage', 
                              global_min=None, global_max=None, threshold=None):
    """
    Create flatmap visualization for group-level data
    """
    # Load your group-average MNI-space image
    img_file = f'{results_dir}/group_averages/{roi}_{hemi}_{analysis_type}_avg.nii.gz'
    print(f"Loading: {img_file}")
    
    # Load and examine data range
    img = nib.load(img_file)
    data = img.get_fdata()
    
    # Print data statistics for examination
    print(f"\nData statistics for {roi}_{hemi}:")
    print(f"Min value: {np.nanmin(data)}")
    print(f"Max value: {np.nanmax(data)}")
    print(f"Mean value: {np.nanmean(data)}")
    print(f"Standard deviation: {np.nanstd(data)}")
    
    try:
        # Create the volume object with the data using global values
        vol = cortex.Volume(
            img_file,
            subject,
            'atlas_2mm',
            cmap='YlOrRd_r',  # Red-Yellow colormap
            vmin=global_min,  # Start at 0
            vmax=global_max   # Use global max
        )
        
        # Apply thresholding to make low values transparent
        vol.data[np.abs(vol.data) <= threshold] = np.nan
        
        # Create figure directory if it doesn't exist
        figure_dir = f'{results_dir}/flatmaps'
        os.makedirs(figure_dir, exist_ok=True)
        
        # Create the flatmap
        fig = cortex.quickflat.make_figure(
            vol,
            with_curvature=True,
            with_rois=True,
            with_colorbar=True
        )
        
        # Save the figure
        output_file = f'{figure_dir}/{roi}_{hemi}_{analysis_type}_flatmap.png'
        cortex.quickflat.make_png(
            output_file,
            vol,
            with_curvature=True,
            dpi=300,
            with_rois=True,
            with_colorbar=True,
            with_labels=True  # This will add labels to the ROIs
        )
        
        # Optional: If you want to view interactively (uncomment if needed)
        #cortex.webgl.show(data=vol)
        
        print(f"Successfully created flatmap for {roi} {hemi}")
        plt.close(fig)  # Clean up the figure
        
    except Exception as e:
        print(f"Error in visualization for {roi} {hemi}: {str(e)}")
        import traceback
        traceback.print_exc()

def get_global_stats(rois, hemispheres, analysis_type='fc'):
    """Get global min, max, and threshold values across all ROIs"""
    all_data = []
    stats = {}
    
    # First pass: collect all data and individual stats
    print("\nCollecting data statistics across all ROIs:")
    for roi in rois:
        for hemi in hemispheres:
            img_file = f'{results_dir}/group_averages/{roi}_{hemi}_{analysis_type}_avg.nii.gz'
            data = nib.load(img_file).get_fdata()
            
            # Store statistics for this ROI
            stats[f"{roi}_{hemi}"] = {
                "min": np.nanmin(data),
                "max": np.nanmax(data),
                "mean": np.nanmean(data),
                "std": np.nanstd(data)
            }
            
            # Print individual stats
            print(f"\nStatistics for {roi}_{hemi}:")
            print(f"Min value: {stats[f'{roi}_{hemi}']['min']}")
            print(f"Max value: {stats[f'{roi}_{hemi}']['max']}")
            print(f"Mean value: {stats[f'{roi}_{hemi}']['mean']}")
            print(f"Standard deviation: {stats[f'{roi}_{hemi}']['std']}")
            
            all_data.append(data)
    
    # Calculate global statistics
    all_data = np.concatenate([d.flatten() for d in all_data])
    global_min = 0  # Start at 0 for red-yellow colormap
    global_max = np.nanmax(all_data)
    global_std = np.nanstd(all_data)
    
    # Use std-based threshold
    #threshold = global_std * 0.5
    threshold = .12
    
    print(f"\nGlobal statistics:")
    print(f"Global min: {global_min}")
    print(f"Global max: {global_max}")
    print(f"Global std: {global_std}")
    print(f"Using threshold: {threshold}")
    
    return global_min, global_max, threshold

        
def main():
    # Print diagnostic information
    print(f"Pycortex database: {cortex.database.default_filestore}")
    print(f"Results directory: {results_dir}")
    
    # Define ROIs and hemispheres
    rois = ['pIPS', 'LO']
    hemispheres = ['left', 'right']
    
    # Get global statistics
    global_min, global_max, threshold = get_global_stats(rois, hemispheres)
    
    # Create both flatmap and inflated visualizations
    for roi in rois:
        for hemi in hemispheres:
            print(f"\nProcessing {roi} {hemi}")
            # Create flatmap
            create_flatmap_visualization(roi, hemi, global_min=global_min, 
                                      global_max=global_max, threshold=threshold)

if __name__ == "__main__":
    main()