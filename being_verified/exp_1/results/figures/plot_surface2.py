'''
module load freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh
'''
import os
import numpy as np
import nibabel as nib
from nilearn import plotting
import matplotlib.pyplot as plt

def visualize_mgz_on_surface(mgz_file, output_file):
    """
    Visualize MGZ file on FreeSurfer inflated surface
    """
    # Load the data
    print(f"Loading data from: {mgz_file}")
    data = nib.load(mgz_file).get_fdata()
    
    # Load FreeSurfer inflated surfaces
    surf_dir = os.path.join(os.environ.get('FREESURFER_HOME'), 'subjects/fsaverage/surf')
    lh_inflated = os.path.join(surf_dir, 'lh.inflated')
    rh_inflated = os.path.join(surf_dir, 'rh.inflated')
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot left hemisphere
    plotting.plot_surf_stat_map(lh_inflated,
                              stat_map=data,
                              hemi='left',
                              view='lateral',
                              colorbar=True,
                              threshold=0.12,
                              bg_map=None,
                              axes=ax1,
                              cmap='RdBu_r')
    ax1.set_title('Left Hemisphere')
    
    # Plot right hemisphere
    plotting.plot_surf_stat_map(rh_inflated,
                              stat_map=data,
                              hemi='right',
                              view='lateral',
                              colorbar=True,
                              threshold=0.12,
                              bg_map=None,
                              axes=ax2,
                              cmap='RdBu_r')
    ax2.set_title('Right Hemisphere')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"File saved: {output_file}")

# Set up directories
fs_dir = '/user_data/csimmon2/git_repos/ptoc/results/freesurfer_space'
out_dir = '/user_data/csimmon2/git_repos/ptoc/being_verified/exp_1/results/figures'

# Make sure output directory exists
os.makedirs(out_dir, exist_ok=True)

# Try one file first
test_file = os.path.join(fs_dir, 'group_pIPS_left_fc_avg_fs.mgz')
test_output = os.path.join(out_dir, 'test_freesurfer_style.png')

print("\nTesting with a single file first...")
try:
    visualize_mgz_on_surface(test_file, test_output)
except Exception as e:
    print(f"Error during visualization: {str(e)}")
    import traceback
    traceback.print_exc()

print("\nChecking output directory...")
if os.path.exists(test_output):
    print(f"Test file created successfully: {test_output}")
else:
    print("Test file was not created")