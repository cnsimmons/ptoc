# Prepare overlap data for surface visualization with PyCortex
import os
import numpy as np
import nibabel as nib
from nilearn import datasets
from nilearn.image import resample_to_img
import argparse

# Define study directories
results_dir = '/user_data/csimmon2/git_repos/ptoc/results'
group_out_dir = f'{results_dir}/group_averages'
surface_out_dir = f'{results_dir}/surface_maps'

# Create surface output directory if it doesn't exist
os.makedirs(surface_out_dir, exist_ok=True)

def prepare_overlap_data(analysis_type='fc', hemisphere='both', subject='fsaverage', xfmname='atlas_2mm'):
    """
    Prepare overlap data for surface visualization using PyCortex.
    
    Parameters:
    -----------
    analysis_type : str
        Analysis type to visualize ('fc' or 'ppi')
    hemisphere : str
        Which hemisphere to visualize ('left', 'right', or 'both')
    subject : str
        Subject ID for PyCortex
    xfmname : str
        Transform name for PyCortex
    """
    print(f"Preparing {analysis_type.upper()} analysis for surface visualization...")
    
    # Define colors to match the violin plot
    pips_color = '#4ac0c0'  # Teal for pIPS
    lo_color = '#ff9b83'    # Peach for LO
    overlap_color = '#9467bd'  # Purple for overlap
    
    if hemisphere == 'both':
        hemispheres = ['left', 'right']
    else:
        hemispheres = [hemisphere]
    
    # Load MNI template
    template = datasets.load_mni152_template(resolution=1)
    
    # Load images for specified hemispheres for each ROI
    pips_imgs = []
    lo_imgs = []
    
    for hemi in hemispheres:
        # Define file paths
        pips_img_path = f'{group_out_dir}/pIPS_{hemi}_{analysis_type}_thresh.nii.gz'
        lo_img_path = f'{group_out_dir}/LO_{hemi}_{analysis_type}_thresh.nii.gz'
        
        # Check if files exist
        if not os.path.exists(pips_img_path) or not os.path.exists(lo_img_path):
            print(f"Warning: Files not found for {hemi} hemisphere, {analysis_type}. Skipping.")
            continue
        
        # Load images
        pips_imgs.append(nib.load(pips_img_path))
        lo_imgs.append(nib.load(lo_img_path))
    
    # Skip if we don't have any images
    if len(pips_imgs) == 0 or len(lo_imgs) == 0:
        print(f"Warning: No data found for {analysis_type}. Skipping.")
        return None
    
    # Get the affine and header from the first image
    affine = pips_imgs[0].affine
    header = pips_imgs[0].header
    
    # Create average maps for each ROI if multiple hemispheres
    if len(pips_imgs) > 1:
        pips_avg_data = (pips_imgs[0].get_fdata() + pips_imgs[1].get_fdata()) / 2
        lo_avg_data = (lo_imgs[0].get_fdata() + lo_imgs[1].get_fdata()) / 2
    else:
        pips_avg_data = pips_imgs[0].get_fdata()
        lo_avg_data = lo_imgs[0].get_fdata()
    
    # Create binary maps for computation
    pips_binary = (pips_avg_data > 0).astype(int)
    lo_binary = (lo_avg_data > 0).astype(int)
    
    # Create overlap map (1=pIPS only, 2=LO only, 3=overlap)
    overlap_data = np.zeros_like(pips_binary)
    overlap_data[pips_binary == 1] = 1  # pIPS only
    overlap_data[lo_binary == 1] = 2    # LO only
    overlap_data[(pips_binary == 1) & (lo_binary == 1)] = 3  # Overlap
    
    # Create the overlap-only map as binary (1 where overlap occurs, 0 elsewhere)
    overlap_only_data = np.zeros_like(overlap_data)
    overlap_only_data[(pips_binary == 1) & (lo_binary == 1)] = 1
    
    # Create NIfTI images
    overlap_img = nib.Nifti1Image(overlap_data, affine, header)
    overlap_only_img = nib.Nifti1Image(overlap_only_data, affine, header)
    pips_only_img = nib.Nifti1Image(pips_binary * (1 - overlap_only_data), affine, header)
    lo_only_img = nib.Nifti1Image(lo_binary * (1 - overlap_only_data), affine, header)
    
    # Resample images to match template resolution for better visualization
    overlap_resampled = resample_to_img(overlap_img, template)
    overlap_only_resampled = resample_to_img(overlap_only_img, template)
    pips_only_resampled = resample_to_img(pips_only_img, template)
    lo_only_resampled = resample_to_img(lo_only_img, template)
    
    # Save the resampled NIfTI files for surface visualization
    hemi_label = "_".join(hemispheres) if hemisphere == 'both' else hemisphere
    overlap_file = f'{surface_out_dir}/overlap_{analysis_type}_{hemi_label}.nii.gz'
    overlap_only_file = f'{surface_out_dir}/overlap_only_{analysis_type}_{hemi_label}.nii.gz'
    pips_only_file = f'{surface_out_dir}/pips_only_{analysis_type}_{hemi_label}.nii.gz'
    lo_only_file = f'{surface_out_dir}/lo_only_{analysis_type}_{hemi_label}.nii.gz'
    
    nib.save(overlap_resampled, overlap_file)
    #nib.save(overlap_only_resampled, overlap_only_file)
    #nib.save(pips_only_resampled, pips_only_file)
    #nib.save(lo_only_resampled, lo_only_file)
    
    print(f"Saved files for surface visualization:")
    print(f"  Combined map: {overlap_file}")
    #print(f"  Overlap only: {overlap_only_file}")
    #print(f"  pIPS only: {pips_only_file}")
    #print(f"  LO only: {lo_only_file}")
    
    # Calculate overlap statistics
    total_pips_voxels = np.sum(pips_binary)
    total_lo_voxels = np.sum(lo_binary)
    total_overlap_voxels = np.sum(overlap_only_data)
    
    # Calculate Dice coefficient
    if (total_pips_voxels + total_lo_voxels) > 0:
        dice_coef = 2 * total_overlap_voxels / (total_pips_voxels + total_lo_voxels)
    else:
        dice_coef = 0
        
    # Print statistics
    print(f"\nOverlap Statistics for {analysis_type.upper()}:")
    print(f"Total pIPS voxels: {total_pips_voxels}")
    print(f"Total LO voxels: {total_lo_voxels}")
    print(f"Total overlap voxels: {total_overlap_voxels}")
    print(f"Overlap percentage: {total_overlap_voxels/max(1, total_pips_voxels + total_lo_voxels - total_overlap_voxels)*100:.2f}%")
    print(f"Dice coefficient: {dice_coef:.4f}")
    
    return {
        'overlap': overlap_file,
        'overlap_only': overlap_only_file,
        'pips_only': pips_only_file,
        'lo_only': lo_only_file
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare overlap data for surface visualization')
    parser.add_argument('--analysis', '-a', type=str, choices=['fc', 'ppi', 'both'], default='both',
                      help='Analysis type to prepare (fc, ppi, or both)')
    parser.add_argument('--hemisphere', '-hemi', type=str, choices=['left', 'right', 'both'], default='both',
                      help='Hemisphere to prepare (left, right, or both)')
    
    args = parser.parse_args()
    
    if args.analysis == 'both':
        analysis_types = ['fc', 'ppi']
    else:
        analysis_types = [args.analysis]
    
    for analysis_type in analysis_types:
        prepare_overlap_data(analysis_type=analysis_type, hemisphere=args.hemisphere)
    
    print("\nData preparation complete. You can now download the files to your PC")
    print("and visualize them using the plot_surface_interactive.py script.")