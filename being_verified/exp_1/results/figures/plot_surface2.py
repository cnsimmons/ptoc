import cortex
import nibabel as nib

# Load your data
mgz_file = '/user_data/csimmon2/git_repos/ptoc/results/freesurfer_space/group_LO_right_ppi_avg_fs.mgz'
data = nib.load(mgz_file).get_fdata()

# Create the volume
vol = cortex.Volume(data, 'fsaverage', 'identity')

# Launch the web viewer
cortex.webgl.show(vol, 
                 colormap='RdBu_r',
                 vmin=-0.5,
                 vmax=0.5)