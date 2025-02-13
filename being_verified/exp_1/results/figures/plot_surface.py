import os
import numpy as np
import nibabel as nib
import cortex
import matplotlib.pyplot as plt

def visualize_mgz(mgz_file, output_file):
    """
    Simple function to visualize an MGZ file on inflated surface
    """
    print(f"Loading data from: {mgz_file}")
    data = nib.load(mgz_file).get_fdata()
    
    print("Creating volume...")
    vol = cortex.Volume(data, 'fsaverage', 'identity')
    
    print(f"Generating visualization to: {output_file}")
    
    # Create figure and plot
    fig = plt.figure(figsize=(12, 8))
    _ = cortex.quickshow(vol, 
                        with_rois=True,
                        with_labels=True,
                        colormap='RdBu_r',
                        vmin=-0.5,
                        vmax=0.5)
    
    # Save figure
    plt.savefig(output_file, format='svg', bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"File saved: {output_file}")

# Set up directories
fs_dir = '/user_data/csimmon2/git_repos/ptoc/results/freesurfer_space'
out_dir = '/user_data/csimmon2/git_repos/ptoc/being_verified/exp_1/results/figures'

# Make sure output directory exists
os.makedirs(out_dir, exist_ok=True)

# Define parameters
rois = ['pIPS', 'LO']
hemispheres = ['left', 'right']
analyses = ['fc', 'ppi']

# Process all combinations
for roi in rois:
    for hemi in hemispheres:
        for analysis in analyses:
            input_file = os.path.join(fs_dir, f'group_{roi}_{hemi}_{analysis}_avg_fs.mgz')
            output_file = os.path.join(out_dir, f'{roi}_{hemi}_{analysis}_inflated.svg')
            
            if os.path.exists(input_file):
                print(f"\nProcessing: {input_file}")
                try:
                    visualize_mgz(input_file, output_file)
                except Exception as e:
                    print(f"Error processing {input_file}: {str(e)}")
            else:
                print(f"File not found: {input_file}")

print("\nAll processing complete!")
print("\nFiles created:")
files = [f for f in os.listdir(out_dir) if f.endswith('.svg')]
for f in files:
    print(f"- {f}")