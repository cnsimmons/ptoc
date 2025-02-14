import os
import subprocess
from pathlib import Path

def create_surface_visualization(mgz_file, output_file, hemisphere='lh', view='superior'):
    """
    Create 3D inflated surface visualization using tksurfer
    """
    # Extract base filename without path for TCL script
    base_output = os.path.basename(output_file)
    
    # Create simple TCL script
    tcl_script = f"""
    set_brain_view {view}
    redraw
    save_tiff {output_file}
    exit
    """
    
    # Write TCL script to temporary file
    tcl_file = output_file.replace('.png', '.tcl')
    with open(tcl_file, 'w') as f:
        f.write(tcl_script)
    
    # Run tksurfer
    cmd = [
        'tksurfer',
        'fsaverage',
        hemisphere,
        'inflated',
        '-overlay', mgz_file,
        '-fthresh', '3',
        '-fmid', '4.5',
        '-fmax', '6',
        '-tcl', tcl_file
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Successfully created visualization: {output_file}")
        # Clean up TCL file
        os.remove(tcl_file)
    except subprocess.CalledProcessError as e:
        print(f"Error creating visualization: {e}")
        raise

def main():
    # Directories setup remains the same
    fs_dir = '/user_data/csimmon2/git_repos/ptoc/results/freesurfer_space'
    out_dir = '/user_data/csimmon2/git_repos/ptoc/being_verified/exp_1/results/figures/freesurfer'
    os.makedirs(out_dir, exist_ok=True)
    
    # Parameters remain the same
    rois = ['pIPS', 'LO']
    hemispheres = ['left', 'right']
    analyses = ['fc', 'ppi']
    views = ['superior', 'lateral', 'medial']
    
    # Process all combinations
    for roi in rois:
        for hemi in hemispheres:
            for analysis in analyses:
                input_file = os.path.join(fs_dir, f'group_{roi}_{hemi}_{analysis}_avg_fs.mgz')
                
                if not os.path.exists(input_file):
                    print(f"File not found: {input_file}")
                    continue
                
                for view in views:
                    output_file = os.path.join(
                        out_dir, 
                        f'{roi}_{hemi}_{analysis}_{view}.png'
                    )
                    
                    try:
                        create_surface_visualization(
                            input_file,
                            output_file,
                            hemisphere='lh' if hemi == 'left' else 'rh',
                            view=view
                        )
                    except Exception as e:
                        print(f"Error processing {input_file} for {view} view: {str(e)}")

if __name__ == "__main__":
    main()