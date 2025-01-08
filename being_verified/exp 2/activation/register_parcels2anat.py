'''CHANGE DIRECTORIES!'''
'''before you run
conda activate fmri
module load fsl-6.0.3
'''

curr_dir = '/user_data/csimmon2/git_repos/ptoc'
import sys

#insert current directory to path
sys.path.insert(0,curr_dir)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
from nilearn import image, plotting, datasets, masking
import nibabel as nib
import pdb
import os
import subprocess
from nilearn.datasets import load_mni152_brain_mask, load_mni152_template
import ptoc_params as params 

raw_dir = params.raw_dir  # /lab_data/behrmannlab/vlad/hemispace

# Load sub_info_tool instead of sub_info and filter for spaceloc subjects
sub_info = params.sub_info_tool
sub_info = sub_info[sub_info['sub'].str.contains('spaceloc')]

#left is negative, right is positive
mni = load_mni152_brain_mask()
mni_affine = mni.affine
parcel_mni='/opt/fsl/6.0.3/data/standard/MNI152_T1_2mm_brain.nii.gz' #this is the MNI we use for both julian and mruczek parcels
anat_mni='/opt/fsl/6.0.3/data/standard/MNI152_T1_2mm_brain.nii.gz' #this is the MNI we use for analysis

parcel_root = f"{curr_dir}/roiParcels"
parcel_type = ""

parcels = params.rois  # Using rois from params file

def create_mirror_brain(sub,hemi):
    """
    Creating mirrored brain for patients - likely won't need for spaceloc subjects
    """
    print("creating brain mirror", sub)
    sub_dir = f'{raw_dir}/{sub}/ses-01/'

    #load anat
    anat_mask = image.load_img(f'{sub_dir}/anat/{sub}_ses-01_T1w_brain_mask.nii.gz')
    anat = image.load_img(f'{sub_dir}/anat/{sub}_ses-01_T1w_brain.nii.gz')
    anat = image.get_data(anat)
    affine = anat_mask.affine
    hemi_mask = image.get_data(anat_mask)

    #extract just one hemi
    mid = list((np.array((hemi_mask.shape))/2).astype(int)) #find mid point of image

    hemi_mask[hemi_mask>0] = 1 #ensure to mask all of it
    anat_flip = anat
    anat_mirror = anat
    anat_flip =anat_flip[::-1,:, :]

    if hemi == 'Left':
        hemi_mask[mid[0]:, :, :] = 0 
        anat_mirror[mid[0]:,:,:] = anat_flip[mid[0]:,:,:]
    else:
        hemi_mask[:mid[0], :, :] = 0 
        anat_mirror[:mid[0],:,:] = anat_flip[:mid[0],:,:]

    anat_mirror = nib.Nifti1Image(anat_mirror, affine)  
    hemi_mask = nib.Nifti1Image(hemi_mask, affine)  
    nib.save(hemi_mask,f'{sub_dir}/anat/{sub}_ses-01_T1w_brain_mask_{hemi}.nii.gz')
    nib.save(anat_mirror,f'{sub_dir}/anat/{sub}_ses-01_T1w_brain_mirrored.nii.gz')
    print('mirror saved to', f'{sub_dir}/anat/{sub}_ses-01_T1w_brain_mirrored.nii.gz')

def create_hemi_mask(sub):
    """
    Creating hemispheric masks for control subjects
    """
    print("creating hemisphere mask", sub)
    sub_dir = f'{raw_dir}/{sub}/ses-01/'

    for hemi in ['Left','Right']:
        #load anat
        anat_mask = image.load_img(f'{sub_dir}/anat/{sub}_ses-01_T1w_brain_mask.nii.gz')
        affine = anat_mask.affine
    
        hemi_mask = image.get_data(anat_mask)
        
        #extract just one hemi
        mid = list((np.array((hemi_mask.shape))/2).astype(int)) #find mid point of image

        hemi_mask[hemi_mask>0] = 1 #ensure to mask all of it

        if hemi == 'Left':
            hemi_mask[mid[0]:, :, :] = 0 
        else:
            hemi_mask[:mid[0], :, :] = 0 

        hemi_mask = nib.Nifti1Image(hemi_mask, affine)  
        nib.save(hemi_mask,f'{sub_dir}/anat/{sub}_ses-01_T1w_brain_mask_{hemi}.nii.gz')

def register_mni(sub,group):
    '''
    Register to MNI
    '''
    print('Registering subj to MNI...', sub)
    anat_dir = f'{raw_dir}/{sub}/ses-01/anat/'
    if group == 'patient':
        anat_mirror = f'{anat_dir}/{sub}_ses-01_T1w_brain_mirrored.nii.gz'
    else:
        anat_mirror = f'{anat_dir}/{sub}_ses-01_T1w_brain.nii.gz'
    
    anat = f'{anat_dir}/{sub}_ses-01_T1w_brain.nii.gz'

    #create registration matrix for subject to mni
    bash_cmd = f'flirt -in {anat_mirror} -ref {anat_mni} -omat {anat_dir}/anat2stand.mat -bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12'
    subprocess.run(bash_cmd.split(), check = True)

    #Create mni of subject brain
    bash_cmd = f'flirt -in {anat} -ref {anat_mni} -out {anat_dir}/{sub}_ses-01_T1w_brain_stand.nii.gz -applyxfm -init {anat_dir}/anat2stand.mat -interp trilinear'
    subprocess.run(bash_cmd.split(), check = True)

    #create registration matrix for mni to subject
    bash_cmd = f'flirt -in {parcel_mni} -ref {anat_mirror} -omat {anat_dir}/mni2anat.mat -bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12'
    subprocess.run(bash_cmd.split(), check = True)

def register_parcels(sub, parcel_dir, parcels): 
    """
    Register parcels to subject
    """
    print("Registering parcels for ", sub)
    roi_dir = f'{raw_dir}/{sub}/ses-01/derivatives/rois'  # Updated path
    anat_dir = f'{raw_dir}/{sub}/ses-01/anat/'
    anat = f'{anat_dir}/{sub}_ses-01_T1w_brain.nii.gz'
    os.makedirs(f'{roi_dir}/parcels/', exist_ok=True)

    for rp in parcels:
        roi_parcel = f'{parcel_dir}{rp}.nii.gz'
        bash_cmd = f'flirt -in {roi_parcel} -ref {anat} -out {roi_dir}/parcels/{rp}.nii.gz -applyxfm -init {anat_dir}/mni2anat.mat -interp trilinear'
        subprocess.run(bash_cmd.split(), check = True)

        #binarize
        bash_cmd = f'fslmaths {roi_dir}/parcels/{rp}.nii.gz -bin {roi_dir}/parcels/{rp}.nii.gz'
        subprocess.run(bash_cmd.split(), check = True)
        
        print(f"Registered {rp}")

# Main execution
parcel_dir = f'{parcel_root}/{parcel_type}'

for sub, hemi, group in zip(sub_info['sub'], sub_info['intact_hemi'], sub_info['group']):
    # Don't need sub- prefix check since they already have it in sub_info_tool
    print(sub, hemi, group)
    
    # Create hemisphere masks for controls
    create_hemi_mask(sub)
    
    # Register to MNI
    register_mni(sub, group)
    
    # Register ROI parcels
    register_parcels(sub, parcel_dir, parcels)