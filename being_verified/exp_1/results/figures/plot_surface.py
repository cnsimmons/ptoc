import os
import numpy as np
import nibabel as nib
import cortex

def visualize_mgz(mgz_file, output_file):
    """
    Simple function to visualize an MGZ file on inflated surface
    """
    # Load data
    data = nib.load(mgz_file).get_fdata()
    
    # Create volume - using string 'identity' instead of Transform object
    vol = cortex.Volume(data, 'fsaverage', 'identity')
    
    # Visualize
    cortex.quickshow(vol, 
                    with_rois=True,
                    with_labels=True,
                    save_name=output_file)

# Set up directories
fs_dir = '/user_data/csimmon2/git_repos/ptoc/results/freesurfer_space'
out_dir = '/user_data/csimmon2/git_repos/ptoc/results/surface_visualizations'
os.makedirs(out_dir, exist_ok=True)

# Try with one file first
test_file = os.path.join(fs_dir, 'group_pIPS_left_fc_avg_fs.mgz')
output_file = os.path.join(out_dir, 'test_visualization.svg')

print(f"Processing: {test_file}")
visualize_mgz(test_file, output_file)
print(f"Done. Check: {output_file}")