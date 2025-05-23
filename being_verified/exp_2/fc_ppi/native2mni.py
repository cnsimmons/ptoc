import os
import sys
import subprocess
import pandas as pd

# Set up directories and parameters
curr_dir = '/user_data/csimmon2/git_repos/ptoc'
study_dir = "/lab_data/behrmannlab/vlad/ptoc"
raw_dir = "/lab_data/behrmannlab/vlad/hemispace"
results_dir = "/user_data/csimmon2/git_repos/ptoc/results/tools"
mni_brain = os.path.join(os.environ['FSLDIR'], "data/standard/MNI152_T1_2mm_brain.nii.gz")

# subjects
sub_info = pd.read_csv(f'{curr_dir}/sub_info_tool.csv')
subs = sub_info[sub_info['exp'] == 'spaceloc']['sub'].tolist()

# Define conditions
conditions = ['tools','nontools']

for sub in subs:
    print(f"Processing subject: {sub}")
    sub_dir = f"{study_dir}/{sub}/ses-01"
    out_dir = f"{sub_dir}/derivatives"
    anat_brain = f"{raw_dir}/{sub}/ses-01/anat/{sub}_ses-01_T1w_brain.nii.gz"

    # Check if anatomical image exists
    if not os.path.isfile(anat_brain):
        print(f"Anatomical image not found for {sub}. Skipping...")
        continue
    
    # Create MNI output directories if they don't exist
    mni_fc_dir = os.path.join(out_dir, 'fc', 'mni')
    mni_ppi_dir = os.path.join(out_dir, 'ppi', 'mni')
    os.makedirs(mni_fc_dir, exist_ok=True)
    os.makedirs(mni_ppi_dir, exist_ok=True)

    # Check if transformation matrix exists, generate only if needed
    anat2mni_mat = f"{out_dir}/fc/anat2mni.mat"
    if not os.path.isfile(anat2mni_mat):
        print(f"Generating transformation matrix for {sub}")
        os.makedirs(os.path.dirname(anat2mni_mat), exist_ok=True)
        subprocess.run([
            'flirt',
            '-in', anat_brain,
            '-ref', mni_brain,
            '-omat', anat2mni_mat,
            '-bins', '256',
            '-cost', 'corratio',
            '-searchrx', '-90', '90',
            '-searchry', '-90', '90',
            '-searchrz', '-90', '90',
            '-dof', '12'
        ], check=True)
    else:
        print(f"Using existing transformation matrix for {sub}")

    # Loop through ROIs and hemispheres
    rois = ['pIPS','LO']
    hemispheres = ['left', 'right']
    
    # Process tools condition (original files without condition in name)
    for roi in rois:
        for hemi in hemispheres:
            # FC to MNI - tools condition (original files)
            fc_native = f"{out_dir}/fc/{sub}_{roi}_{hemi}_ToolLoc_fc.nii.gz"
            fc_mni = f"{mni_fc_dir}/{sub}_{roi}_{hemi}_ToolLoc_fc_mni.nii.gz"
            
            if os.path.isfile(fc_native) and not os.path.isfile(fc_mni):
                print(f"Registering FC for {sub}, ROI {roi}, Hemisphere {hemi} to MNI space")
                subprocess.run([
                    'flirt',
                    '-in', fc_native,
                    '-ref', mni_brain,
                    '-out', fc_mni,
                    '-applyxfm',
                    '-init', anat2mni_mat,
                    '-interp', 'trilinear'
                ], check=True)
            elif os.path.isfile(fc_mni):
                print(f"FC file for {sub}, ROI {roi}, Hemisphere {hemi} already registered to MNI space")
            else:
                print(f"FC file not found for {sub}, ROI {roi}, Hemisphere {hemi}")
        
            # PPI to MNI - tools condition (original files)
            ppi_native = f"{out_dir}/ppi/{sub}_{roi}_{hemi}_ToolLoc_ppi.nii.gz"
            ppi_mni = f"{mni_ppi_dir}/{sub}_{roi}_{hemi}_ToolLoc_ppi_mni.nii.gz"
            
            if os.path.isfile(ppi_native) and not os.path.isfile(ppi_mni):
                print(f"Registering PPI for {sub}, ROI {roi}, Hemisphere {hemi} to MNI space")
                subprocess.run([
                    'flirt',
                    '-in', ppi_native,
                    '-ref', mni_brain,
                    '-out', ppi_mni,
                    '-applyxfm',
                    '-init', anat2mni_mat,
                    '-interp', 'trilinear'
                ], check=True)
            elif os.path.isfile(ppi_mni):
                print(f"PPI file for {sub}, ROI {roi}, Hemisphere {hemi} already registered to MNI space")
            else:
                print(f"PPI file not found for {sub}, ROI {roi}, Hemisphere {hemi}")
    
    # Process nontools condition
    for roi in rois:
        for hemi in hemispheres:
            # FC to MNI - nontools condition
            nontools_fc_native = f"{out_dir}/fc/{sub}_{roi}_{hemi}_nontools_ToolLoc_fc.nii.gz"
            nontools_fc_mni = f"{mni_fc_dir}/{sub}_{roi}_{hemi}_nontools_ToolLoc_fc_mni.nii.gz"
            
            if os.path.isfile(nontools_fc_native) and not os.path.isfile(nontools_fc_mni):
                print(f"Registering nontools FC for {sub}, ROI {roi}, Hemisphere {hemi} to MNI space")
                subprocess.run([
                    'flirt',
                    '-in', nontools_fc_native,
                    '-ref', mni_brain,
                    '-out', nontools_fc_mni,
                    '-applyxfm',
                    '-init', anat2mni_mat,
                    '-interp', 'trilinear'
                ], check=True)
            elif os.path.isfile(nontools_fc_mni):
                print(f"Nontools FC file for {sub}, ROI {roi}, Hemisphere {hemi} already registered to MNI space")
            else:
                print(f"Nontools FC file not found for {sub}, ROI {roi}, Hemisphere {hemi}")
        
            # PPI to MNI - nontools condition
            nontools_ppi_native = f"{out_dir}/ppi/{sub}_{roi}_{hemi}_nontools_ToolLoc_ppi.nii.gz"
            nontools_ppi_mni = f"{mni_ppi_dir}/{sub}_{roi}_{hemi}_nontools_ToolLoc_ppi_mni.nii.gz"
            
            if os.path.isfile(nontools_ppi_native) and not os.path.isfile(nontools_ppi_mni):
                print(f"Registering nontools PPI for {sub}, ROI {roi}, Hemisphere {hemi} to MNI space")
                subprocess.run([
                    'flirt',
                    '-in', nontools_ppi_native,
                    '-ref', mni_brain,
                    '-out', nontools_ppi_mni,
                    '-applyxfm',
                    '-init', anat2mni_mat,
                    '-interp', 'trilinear'
                ], check=True)
            elif os.path.isfile(nontools_ppi_mni):
                print(f"Nontools PPI file for {sub}, ROI {roi}, Hemisphere {hemi} already registered to MNI space")
            else:
                print(f"Nontools PPI file not found for {sub}, ROI {roi}, Hemisphere {hemi}")
                
    # Process tools vs nontools contrast
    for roi in rois:
        for hemi in hemispheres:
            # PPI to MNI - tools vs nontools contrast
            tools_vs_nontools_ppi_native = f"{out_dir}/ppi/{sub}_{roi}_{hemi}_tools_vs_nontools_ToolLoc_ppi.nii.gz"
            tools_vs_nontools_ppi_mni = f"{mni_ppi_dir}/{sub}_{roi}_{hemi}_tools_vs_nontools_ToolLoc_ppi_mni.nii.gz"
            
            if os.path.isfile(tools_vs_nontools_ppi_native) and not os.path.isfile(tools_vs_nontools_ppi_mni):
                print(f"Registering tools vs nontools PPI for {sub}, ROI {roi}, Hemisphere {hemi} to MNI space")
                subprocess.run([
                    'flirt',
                    '-in', tools_vs_nontools_ppi_native,
                    '-ref', mni_brain,
                    '-out', tools_vs_nontools_ppi_mni,
                    '-applyxfm',
                    '-init', anat2mni_mat,
                    '-interp', 'trilinear'
                ], check=True)
            elif os.path.isfile(tools_vs_nontools_ppi_mni):
                print(f"Tools vs nontools PPI file for {sub}, ROI {roi}, Hemisphere {hemi} already registered to MNI space")
            else:
                print(f"Tools vs nontools PPI file not found for {sub}, ROI {roi}, Hemisphere {hemi}")

    print(f"Conversion to MNI space completed for {sub}.")

'''
import os
import sys
import subprocess
import pandas as pd

# Set up directories and parameters
curr_dir = '/user_data/csimmon2/git_repos/ptoc'
study_dir = "/lab_data/behrmannlab/vlad/ptoc"
raw_dir = "/lab_data/behrmannlab/vlad/hemispace"
results_dir = "/user_data/csimmon2/git_repos/ptoc/results/tools"
mni_brain = os.path.join(os.environ['FSLDIR'], "data/standard/MNI152_T1_2mm_brain.nii.gz")

# subjects
sub_info = pd.read_csv(f'{curr_dir}/sub_info_tool.csv')
subs = sub_info[sub_info['exp'] == 'spaceloc']['sub'].tolist()

# Define conditions
conditions = ['tools','nontools']

for sub in subs:
    print(f"Processing subject: {sub}")
    sub_dir = f"{study_dir}/{sub}/ses-01"
    out_dir = f"{sub_dir}/derivatives"
    anat_brain = f"{raw_dir}/{sub}/ses-01/anat/{sub}_ses-01_T1w_brain.nii.gz"

    # Check if anatomical image exists
    if not os.path.isfile(anat_brain):
        print(f"Anatomical image not found for {sub}. Skipping...")
        continue
    
    # Create MNI output directories if they don't exist
    mni_fc_dir = os.path.join(out_dir, 'fc', 'mni')
    mni_ppi_dir = os.path.join(out_dir, 'ppi', 'mni')
    os.makedirs(mni_fc_dir, exist_ok=True)
    os.makedirs(mni_ppi_dir, exist_ok=True)

    # Always generate a new transformation matrix
    anat2mni_mat = f"{out_dir}/fc/anat2mni.mat"
    print(f"Generating transformation matrix for {sub}")
    subprocess.run([
        'flirt',
        '-in', anat_brain,
        '-ref', mni_brain,
        '-omat', anat2mni_mat,
        '-bins', '256',
        '-cost', 'corratio',
        '-searchrx', '-90', '90',
        '-searchry', '-90', '90',
        '-searchrz', '-90', '90',
        '-dof', '12'
    ], check=True)

    # Loop through ROIs and hemispheres
    rois = ['pIPS','LO']
    hemispheres = ['left', 'right']
    
    # Process tools condition (original files without condition in name)
    for roi in rois:
        for hemi in hemispheres:
            # FC to MNI - tools condition (original files)
            fc_native = f"{out_dir}/fc/{sub}_{roi}_{hemi}_ToolLoc_fc.nii.gz"
            fc_mni = f"{mni_fc_dir}/{sub}_{roi}_{hemi}_ToolLoc_fc_mni.nii.gz"
            
            if os.path.isfile(fc_native):
                print(f"Registering FC for {sub}, ROI {roi}, Hemisphere {hemi} to MNI space")
                subprocess.run([
                    'flirt',
                    '-in', fc_native,
                    '-ref', mni_brain,
                    '-out', fc_mni,
                    '-applyxfm',
                    '-init', anat2mni_mat,
                    '-interp', 'trilinear'
                ], check=True)
            else:
                print(f"FC file not found for {sub}, ROI {roi}, Hemisphere {hemi}")
        
            # PPI to MNI - tools condition (original files)
            ppi_native = f"{out_dir}/ppi/{sub}_{roi}_{hemi}_ToolLoc_ppi.nii.gz"
            ppi_mni = f"{mni_ppi_dir}/{sub}_{roi}_{hemi}_ToolLoc_ppi_mni.nii.gz"
            
            if os.path.isfile(ppi_native):
                print(f"Registering PPI for {sub}, ROI {roi}, Hemisphere {hemi} to MNI space")
                subprocess.run([
                    'flirt',
                    '-in', ppi_native,
                    '-ref', mni_brain,
                    '-out', ppi_mni,
                    '-applyxfm',
                    '-init', anat2mni_mat,
                    '-interp', 'trilinear'
                ], check=True)
            else:
                print(f"PPI file not found for {sub}, ROI {roi}, Hemisphere {hemi}")
    
    # Process nontools condition
    for roi in rois:
        for hemi in hemispheres:
            # FC to MNI - nontools condition
            nontools_fc_native = f"{out_dir}/fc/{sub}_{roi}_{hemi}_nontools_ToolLoc_fc.nii.gz"
            nontools_fc_mni = f"{mni_fc_dir}/{sub}_{roi}_{hemi}_nontools_ToolLoc_fc_mni.nii.gz"
            
            if os.path.isfile(nontools_fc_native):
                print(f"Registering nontools FC for {sub}, ROI {roi}, Hemisphere {hemi} to MNI space")
                subprocess.run([
                    'flirt',
                    '-in', nontools_fc_native,
                    '-ref', mni_brain,
                    '-out', nontools_fc_mni,
                    '-applyxfm',
                    '-init', anat2mni_mat,
                    '-interp', 'trilinear'
                ], check=True)
            else:
                print(f"Nontools FC file not found for {sub}, ROI {roi}, Hemisphere {hemi}")
        
            # PPI to MNI - nontools condition
            nontools_ppi_native = f"{out_dir}/ppi/{sub}_{roi}_{hemi}_nontools_ToolLoc_ppi.nii.gz"
            nontools_ppi_mni = f"{mni_ppi_dir}/{sub}_{roi}_{hemi}_nontools_ToolLoc_ppi_mni.nii.gz"
            
            if os.path.isfile(nontools_ppi_native):
                print(f"Registering nontools PPI for {sub}, ROI {roi}, Hemisphere {hemi} to MNI space")
                subprocess.run([
                    'flirt',
                    '-in', nontools_ppi_native,
                    '-ref', mni_brain,
                    '-out', nontools_ppi_mni,
                    '-applyxfm',
                    '-init', anat2mni_mat,
                    '-interp', 'trilinear'
                ], check=True)
            else:
                print(f"Nontools PPI file not found for {sub}, ROI {roi}, Hemisphere {hemi}")
                
    # Process tools vs nontools contrast
    for roi in rois:
        for hemi in hemispheres:
            # PPI to MNI - tools vs nontools contrast
            tools_vs_nontools_ppi_native = f"{out_dir}/ppi/{sub}_{roi}_{hemi}_tools_vs_nontools_ToolLoc_ppi.nii.gz"
            tools_vs_nontools_ppi_mni = f"{mni_ppi_dir}/{sub}_{roi}_{hemi}_tools_vs_nontools_ToolLoc_ppi_mni.nii.gz"
            
            if os.path.isfile(tools_vs_nontools_ppi_native):
                print(f"Registering tools vs nontools PPI for {sub}, ROI {roi}, Hemisphere {hemi} to MNI space")
                subprocess.run([
                    'flirt',
                    '-in', tools_vs_nontools_ppi_native,
                    '-ref', mni_brain,
                    '-out', tools_vs_nontools_ppi_mni,
                    '-applyxfm',
                    '-init', anat2mni_mat,
                    '-interp', 'trilinear'
                ], check=True)
            else:
                print(f"Tools vs nontools PPI file not found for {sub}, ROI {roi}, Hemisphere {hemi}")

    print(f"Conversion to MNI space completed for {sub}.")
'''